"""Segmentacion semantica de las lineas de la pista (U-Net + ResNet34 preentrenado).

Por que segmentacion y no mas keypoints. La via de pose de YOLO esta medida y
tiene dos problemas estructurales que la segmentacion no tiene:

  - de los 56 keypoints solo ~14 caen dentro del frame (mediana medida sobre las
    1093 imagenes etiquetadas), asi que el 76% de la salida no recibe gradiente
  - la caja del rink gatea todo: a conf 0.25 se pierde el 42% de los frames de
    val sin que los keypoints tengan la culpa

Ademas coincide con la literatura (Rink-Agnostic, Waterloo MMSports 2023;
boundary-aware CVIU 2025): las vias que mejor generalizan entre pabellones
segmentan lineas y resuelven la homografia despues, en vez de regresionar puntos
con nombre. Y el dato ya existe: sportcal/lab/hockey/make_line_masks.py genero 1049
mascaras sin anotar nada a mano.

Detalles que NO son opcionales aqui:

  - DESBALANCE 58:1. Las lineas son el 1.68% del pixel (y una clase concreta
    llega a ser 1 de cada 2173). Con cross-entropy plana el modelo predice
    "todo fondo" y acierta el 97.9%. Se usa CE con pesos por clase
    (inverso de la raiz de la frecuencia) + Dice sobre las clases de linea.
  - IGNORE=255. make_line_masks marca asi los pixeles donde la homografia de
    origen no era fiable; la loss los salta (`ignore_index`).
  - FLIP HORIZONTAL con remapeo de clases. Voltear la imagen intercambia las
    zonas A y B pero NO el eje lo/hi (el espejo es en x, y se conserva; es la
    misma logica que MIRROR/flip_idx para los keypoints, ya verificada). Sin el
    remapeo, el flip enseña etiquetas equivocadas.
  - U-Net y no el DeepLabV3 de torchvision: las lineas son de 3-5 px y las
    conexiones skip preservan esa resolucion. El DeepLabV3 de torchvision
    (sin decoder "+") saca a 1/8 y las difumina.

    python -m sportcal.lab.hockey.train_lines_seg
    python -m sportcal.lab.hockey.train_lines_seg --epochs 80 --batch 4 --imgsz 1024x576
"""
import os

import argparse
import copy
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34

from sportcal.paths import ROOT

from sportcal.lab.hockey import make_line_masks as MM

NCLS = 12
IGNORE = 255

# Pares (mascara, imagenes) -- las mascaras llevan el mismo stem que su imagen.
# El dataset sintetico (gen_synthetic_lines.py) SOLO entra en train -- val debe
# medir generalizacion sobre video real, nunca sobre lo que la propia sintesis
# genero (si no, IoU/z de val quedarian artificialmente altos e inutiles).
REAL_JOBS = [("hockeyrink_lines", "hockeyrink"), ("hockeyrink_nhl_lines", "hockeyrink_nhl")]
SYNTH_JOBS = [("hockeyrink_synth_lines", "hockeyrink_synth")]

# Al voltear en horizontal la zona A pasa a ser la B, pero lo/hi (que es el eje y)
# se conserva. Verificado con la misma geometria que MIRROR en rink.py.
FLIP_CLS = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 6, 7: 7,
            8: 10, 9: 11, 10: 8, 11: 9, IGNORE: IGNORE}

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------ datos

class LineasDataset(Dataset):
    def __init__(self, split, imgsz, augmenta, jobs, apariencia=None):
        self.apariencia = apariencia
        self.items = []
        self.synth = []
        synth_names = {m for m, _ in SYNTH_JOBS}
        for mask_ds, img_ds in jobs:
            for m in sorted((ROOT / "datasets" / mask_ds / split).glob("*.png")):
                im = ROOT / "datasets" / img_ds / "images" / split / (m.stem + ".jpg")
                if im.exists():
                    self.items.append((im, m))
                    self.synth.append(mask_ds in synth_names)
        self.w, self.h = imgsz
        self.augmenta = augmenta

    def __len__(self):
        return len(self.items)

    def sample_weights(self):
        """Pesos para WeightedRandomSampler: real y sintetico pesan lo mismo por epoch.

        Sin esto, con miles de imagenes sinteticas y solo cientos reales, la red
        pasaria la mayor parte del entrenamiento viendo solo el render sintetico
        y el ajuste fino al dominio real (lo que de verdad se mide en val) se
        diluiria.
        """
        synth = np.array(self.synth)
        n_real, n_synth = int((~synth).sum()), int(synth.sum())
        w = np.ones(len(synth))
        if n_real:
            w[~synth] = 1.0 / n_real
        if n_synth:
            w[synth] = 1.0 / n_synth
        return w

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.imread(str(ip))
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        # NEAREST obligatorio: interpolar una mascara de indices inventa clases
        msk = cv2.resize(msk, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        if self.augmenta:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                msk = msk[:, ::-1].copy()
                remap = np.arange(256, dtype=np.uint8)
                for a, b in FLIP_CLS.items():
                    remap[a] = b
                msk = remap[msk]
            if np.random.rand() < 0.8:          # color: distinto pabellon, distinta luz
                img = img.astype(np.float32)
                img *= np.random.uniform(0.75, 1.25)
                img += np.random.uniform(-20, 20)
                hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            img = np.clip(img, 0, 255).astype(np.uint8)
            if self.apariencia is not None:
                img = self.apariencia(img, msk)

        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(msk.astype(np.int64))


class TargetDataset(Dataset):
    """Frames de video SIN etiquetar (extract_uda_frames.py) para la adaptacion de
    dominio. Solo aumentacion geometrica/color suave: las pseudo-etiquetas las da el
    profesor sobre la MISMA imagen, asi que cualquier transformacion es consistente."""

    def __init__(self, imgsz):
        self.files = sorted((ROOT / "datasets" / "uda_target" / "images" / "train").glob("*.jpg"))
        self.w, self.h = imgsz

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.resize(cv2.imread(str(self.files[i])), (self.w, self.h), interpolation=cv2.INTER_AREA)
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


def mascara_mic(x, parche=64, ratio=0.5):
    """MIC (Hoyer et al. 2023): tapa parches aleatorios de la imagen objetivo para
    que el alumno tenga que inferir contexto -- las lineas tapadas se deducen del
    resto de la pista. Poner a 0 en espacio normalizado = color medio de ImageNet."""
    b, _, h, w = x.shape
    g = (torch.rand(b, 1, (h + parche - 1) // parche, (w + parche - 1) // parche, device=x.device) < ratio).float()
    g = Fn.interpolate(g, size=(h, w), mode="nearest")
    return x * (1.0 - g)


@torch.no_grad()
def ema_update(teacher, student, alpha):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(alpha).add_(ps.detach(), alpha=1.0 - alpha)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


# ------------------------------------------------------------------ modelo

class Bloque(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))

    def forward(self, x):
        return self.f(x)


class UNetResNet34(nn.Module):
    """U-Net con encoder ResNet34 de ImageNet. Preentrenado importa: con 918
    imagenes de entrenamiento, partir de cero no llega."""

    def __init__(self, ncls=NCLS):
        super().__init__()
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # 64,  1/2
        self.pool = r.maxpool
        self.e1, self.e2, self.e3, self.e4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.d4 = Bloque(512 + 256, 256)
        self.d3 = Bloque(256 + 128, 128)
        self.d2 = Bloque(128 + 64, 64)
        self.d1 = Bloque(64 + 64, 32)
        self.final = nn.Sequential(Bloque(32, 16), nn.Conv2d(16, ncls, 1))

    @staticmethod
    def _up(x, skip):
        x = Fn.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], 1)

    def forward(self, x):
        s0 = self.stem(x)            # 1/2
        s1 = self.e1(self.pool(s0))  # 1/4
        s2 = self.e2(s1)             # 1/8
        s3 = self.e3(s2)             # 1/16
        s4 = self.e4(s3)             # 1/32
        d = self.d4(self._up(s4, s3))
        d = self.d3(self._up(d, s2))
        d = self.d2(self._up(d, s1))
        d = self.d1(self._up(d, s0))
        d = Fn.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.final(d)


# ------------------------------------------------------------------ loss

def pesos_por_clase(loader, max_batches=40):
    """Inverso de la raiz de la frecuencia. La raiz es a proposito: el inverso
    puro daria peso 2000+ a la linea azul B y desestabiliza el entrenamiento."""
    cuenta = torch.zeros(NCLS, dtype=torch.float64)
    for i, (_, y) in enumerate(loader):
        v = y[y != IGNORE]
        cuenta += torch.bincount(v.flatten(), minlength=NCLS).double()
        if i + 1 >= max_batches:
            break
    frec = cuenta / cuenta.sum()
    w = 1.0 / torch.sqrt(frec.clamp(min=1e-8))
    return (w / w.mean()).float()


def dice_lineas(logits, y, eps=1.0):
    """Dice sobre las clases de linea (sin el fondo). Maneja el desbalance por
    construccion: mide solape relativo, no aciertos absolutos."""
    valido = y != IGNORE
    if valido.sum() == 0:
        return logits.sum() * 0.0
    p = torch.softmax(logits, 1)
    y_ = torch.where(valido, y, torch.zeros_like(y))
    oh = Fn.one_hot(y_, NCLS).permute(0, 3, 1, 2).float() * valido.unsqueeze(1)
    p = p * valido.unsqueeze(1)
    inter = (p[:, 1:] * oh[:, 1:]).sum((0, 2, 3))
    suma = p[:, 1:].sum((0, 2, 3)) + oh[:, 1:].sum((0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (suma + eps)).mean()


# ------------------------------------------------------------------ metrica

@torch.no_grad()
def evalua(model, loader, device):
    model.eval()
    inter = torch.zeros(NCLS, dtype=torch.float64)
    union = torch.zeros(NCLS, dtype=torch.float64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        valido = y != IGNORE
        for c in range(NCLS):
            pc, yc = (pred == c) & valido, (y == c) & valido
            inter[c] += (pc & yc).sum().item()
            union[c] += (pc | yc).sum().item()
    iou = (inter / union.clamp(min=1)).numpy()
    presente = union.numpy() > 0
    return iou, presente


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", default="1024x576", help="ancho x alto, multiplos de 32")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--salida", default=str(ROOT / "runs" / "lineas_seg"))
    ap.add_argument("--synth", dest="synth", action="store_true", default=True,
                    help="incluir datasets/hockeyrink_synth_lines en train (por defecto si existe)")
    ap.add_argument("--no-synth", dest="synth", action="store_false")
    ap.add_argument("--init", default=str(ROOT / "runs" / "lineas_seg" / "best.pt"),
                    help="pesos de partida si existen (por defecto el best.pt del run anterior); "
                         "'' para arrancar de ImageNet puro")
    ap.add_argument("--aug-apariencia", action="store_true",
                    help="copy-paste de jugadores reales + logo augmentation (appearance_aug.py)")
    ap.add_argument("--uda", action="store_true",
                    help="adaptacion de dominio MIC: profesor EMA + pseudo-etiquetas sobre "
                         "datasets/uda_target (correr antes extract_uda_frames.py)")
    ap.add_argument("--uda-batch", type=int, default=2)
    ap.add_argument("--uda-lambda", type=float, default=0.5)
    ap.add_argument("--uda-thr-linea", type=float, default=0.7, help="confianza minima para pseudo-etiqueta de linea")
    ap.add_argument("--uda-thr-fondo", type=float, default=0.95, help="confianza minima para pseudo-etiqueta de fondo")
    args = ap.parse_args()

    w, h = (int(v) for v in args.imgsz.lower().split("x"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    tiene_synth = (ROOT / "datasets" / SYNTH_JOBS[0][0] / "train").exists()
    usa_synth = args.synth and tiene_synth
    tr_jobs = REAL_JOBS + (SYNTH_JOBS if usa_synth else [])
    apariencia = None
    if args.aug_apariencia:
        from sportcal.lab.hockey.appearance_aug import AppearanceAug
        apariencia = AppearanceAug()
        print("aumentacion de apariencia: {} recortes de jugadores reales".format(len(apariencia.crops)))
    tr = LineasDataset("train", (w, h), augmenta=True, jobs=tr_jobs, apariencia=apariencia)
    va = LineasDataset("val", (w, h), augmenta=False, jobs=REAL_JOBS)  # val: SIEMPRE solo real
    n_synth_tr = int(np.sum(tr.synth))
    print("train {} imagenes ({} reales, {} sinteticas)   val {} (solo real)   a {}x{} en {}".format(
        len(tr), len(tr) - n_synth_tr, n_synth_tr, len(va), w, h, device))

    if usa_synth and n_synth_tr:
        sampler = torch.utils.data.WeightedRandomSampler(
            tr.sample_weights().tolist(), num_samples=len(tr), replacement=True)
        dl_tr = DataLoader(tr, batch_size=args.batch, sampler=sampler, num_workers=4,
                           pin_memory=True, drop_last=True, persistent_workers=True)
    else:
        dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4,
                           pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=2,
                       pin_memory=True, persistent_workers=True)

    print("calculando pesos por clase...")
    pesos = pesos_por_clase(dl_tr).to(device)
    print("  " + "  ".join("{}={:.1f}".format(MM.CLASSES[c][:9], pesos[c]) for c in range(NCLS)))

    model = UNetResNet34().to(device)
    init_path = Path(args.init) if args.init else None
    if init_path and init_path.exists():
        ckpt = torch.load(init_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("pesos de partida: {} (epoch {} del run anterior)".format(init_path, ckpt.get("epoch", "?")))
    else:
        print("pesos de partida: ninguno (encoder ImageNet, decoder aleatorio)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    escala = torch.amp.GradScaler(device)
    ce = nn.CrossEntropyLoss(weight=pesos, ignore_index=IGNORE)

    teacher = it_tgt = None
    if args.uda:
        tgt = TargetDataset((w, h))
        if not len(tgt):
            raise SystemExit("sin frames objetivo -- corre antes sportcal/lab/hockey/extract_uda_frames.py")
        dl_tgt = DataLoader(tgt, batch_size=args.uda_batch, shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True, persistent_workers=True)
        it_tgt = iter(dl_tgt)
        teacher = copy.deepcopy(model).eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print("UDA (MIC): {} frames objetivo sin etiquetar, lambda={}, umbrales linea/fondo {}/{}".format(
            len(tgt), args.uda_lambda, args.uda_thr_linea, args.uda_thr_fondo))

    mejor = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, suma, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device):
                logits = model(x)
                loss = ce(logits, y) + dice_lineas(logits, y)
                if teacher is not None:
                    try:
                        xt = next(it_tgt)
                    except StopIteration:
                        it_tgt = iter(dl_tgt)
                        xt = next(it_tgt)
                    xt = xt.to(device, non_blocking=True)
                    with torch.no_grad():
                        conf, pl = torch.softmax(teacher(xt).float(), 1).max(1)
                        ok = ((pl > 0) & (conf > args.uda_thr_linea)) | ((pl == 0) & (conf > args.uda_thr_fondo))
                        q = ok.float().mean()                      # ponderacion DAFormer: fraccion fiable
                        pl = torch.where(ok, pl, torch.full_like(pl, IGNORE))
                    lt = ce(model(mascara_mic(xt)).float(), pl)
                    rampa = min(1.0, ep / 3.0)                     # 3 epochs de subida gradual
                    loss = loss + args.uda_lambda * rampa * q * lt
            escala.scale(loss).backward()
            escala.step(opt)
            escala.update()
            if teacher is not None:
                ema_update(teacher, model, 0.999)
            suma += loss.item()
            n += 1
        sched.step()

        iou, presente = evalua(model, dl_va, device)
        lineas = iou[1:][presente[1:]]
        miou = float(lineas.mean()) if len(lineas) else 0.0
        print("ep{:3d}  loss={:.3f}  IoU_lineas={:.3f}  vallas={:.3f}  "
              "gol={:.3f}/{:.3f}  azul={:.3f}/{:.3f}  central={:.3f}  circ={:.3f}  ({:.0f}s)".format(
                  ep, suma / max(n, 1), miou, iou[1], iou[2], iou[3], iou[4], iou[5],
                  iou[6], iou[7], time.time() - t0))

        if miou > mejor:
            mejor = miou
            torch.save({"model": model.state_dict(), "epoch": ep, "iou": iou.tolist(),
                        "imgsz": [w, h]}, out / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": ep, "imgsz": [w, h]}, out / "last.pt")

    print("\nmejor IoU medio de lineas: {:.3f}  ->  {}".format(mejor, out / "best.pt"))


if __name__ == "__main__":
    main()
