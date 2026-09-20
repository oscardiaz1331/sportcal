"""Cabeza aprendida: mapa de probabilidad de segmentacion -> H, sin DLT.

Motivacion completa en CLAUDE.md seccion 7ter. Resumen: seg_to_homography.py
resuelve cada frame con DLT clasico + SVD, que exige dof>=10 (5+
correspondencias no paralelas simultaneas) o se niega a responder -- de ahi
la cobertura del 5-14%. El experimento de reentrenar la segmentacion con mas
datos NHL reales NO subio esa cobertura (de hecho bajo). Esta cabeza ataca el
problema por el otro lado: en vez de resolver algebra por frame sin memoria,
aprende un prior sobre que poses de camara son plausibles, entrenado sobre
las ~1700 imagenes con H de verdad que ya existen (fit_from_label), y da una
H para CUALQUIER frame -- sin puerta de aceptacion, cobertura 100% por
construccion, a costa de precision cuando hay pocas pistas.

Decisiones de diseño (por que, no solo que):

- Entrada: el mapa de 12 canales YA EXISTENTE (softmax de train_lines_seg.py),
  no la imagen cruda -- la red de segmentacion ya hizo el trabajo dificil de
  encontrar QUE es cada pixel; regresionar pose desde RGB crudo reaprenderia
  eso desde cero con muchos menos datos utiles por pixel.
- Salida: pose de camara (rotacion 6D + centro de camara en metros + focal),
  NO los 8 coeficientes de H directamente. H cruda normalizada puede tener
  coeficientes de ordenes de magnitud muy distintos (bug 6 de la seccion 7:
  ~1e14 en un caso real) -- pesimamente condicionado para un gradiente. Pose
  fisica esta acotada y tiene una escala natural consistente.
- Rotacion en representacion 6D (Zhou et al. 2019: "On the Continuity of
  Rotation Representations in Neural Networks"), no Euler ni cuaternion --
  ambas tienen discontinuidades (el mismo giro fisico cerca del limite de
  rango tiene representaciones lejanas en el espacio de salida), que rompen
  el aprendizaje por gradiente. 6D + Gram-Schmidt es continua en todo el
  dominio.
- Perdida de REPROYECCION sobre los 56 puntos de rink.py, no MSE sobre los
  parametros de pose -- exactamente el mismo principio que ya uso
  refine_nonlinear en seg_to_homography.py: lo que importa es el error en
  pixeles, no que tan parecidos sean los numeros crudos. Los 56 puntos se
  supervisan SIEMPRE, esten o no en cuadro -- así la red aprende a
  extrapolar una pose plausible con pocas pistas visibles (la propiedad que
  al DLT le falta por diseño). Huber, no L2 puro: un punto muy fuera de
  cuadro puede dar un error de miles de px y ahogar el gradiente de todo lo
  demas si no se satura (la misma leccion que "f_scale" en el refinamiento
  clasico, seccion 7 bug 5).

    python training/cache_seg_probs.py              # una vez, antes de esto
    python training/homography_head.py --epochs 40
    python training/homography_head.py --eval-only --pesos runs/homography_head/best.pt
"""
import os

_CUDNN_DIR = os.sep.join(("NVIDIA", "CUDNN"))

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if _CUDNN_DIR not in p
)

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import rink  # noqa: E402

NCLS = 12
REF_WIDTH = 1920.0

JOBS = [("hockeyrink", rink.RINK_IIHF), ("hockeyrink_nhl", rink.RINK_NHL)]


# ------------------------------------------------------------------ modelo

def rot6d_to_matrix(x):
    """Zhou et al. 2019 -- x: (B,6) -> R: (B,3,3), ortonormal y continua."""
    a1, a2 = x[:, 0:3], x[:, 3:6]
    b1 = Fnn.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(1, keepdim=True) * b1
    b2 = Fnn.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)   # columnas r1,r2,r3


def pose_to_H(R, C, f, w, h):
    """Pose de camara -> H (metros -> pixeles), plano mundo Z=0.

    R: (B,3,3) rotacion mundo->camara.  C: (B,3) centro de camara en metros.
    f: (B,) focal en px (a la escala nativa de cada frame).  w,h: (B,) tamano
    de imagen nativo. Punto principal fijo en el centro -- simplificacion de
    v1 (ver docstring del modulo); H = K [r1 r2 t], t = -R@C.
    """
    B = R.shape[0]
    t = -torch.bmm(R, C.unsqueeze(-1)).squeeze(-1)                 # (B,3)
    Rt = torch.stack([R[:, :, 0], R[:, :, 1], t], dim=2)           # (B,3,3)
    zero = torch.zeros_like(f)
    one = torch.ones_like(f)
    K = torch.stack([
        torch.stack([f, zero, w / 2], dim=1),
        torch.stack([zero, f, h / 2], dim=1),
        torch.stack([zero, zero, one], dim=1),
    ], dim=1)                                                       # (B,3,3)
    H = torch.bmm(K, Rt)
    # normalizar por H[2,2] preservando el signo -- .clamp(min=1e-6) sobre un
    # valor NEGATIVO (normal, depende de la orientacion de camara) lo sube a
    # +1e-6 en vez de dejarlo negativo, y dividir por eso dispara toda la
    # matriz x~1e7. Bug real, encontrado porque el error de val se quedaba
    # estancado en ~900px sin bajar por mucho que se entrenara.
    denom = H[:, 2, 2].view(-1, 1, 1)
    sign = torch.where(denom >= 0, 1.0, -1.0)
    H = H / (sign * denom.abs().clamp(min=1e-6))
    return H


class PoseHead(nn.Module):
    """CNN pequena sobre el mapa de 12 canales -> pose de camara."""

    def __init__(self, in_ch=NCLS):
        super().__init__()
        # +2 canales de coordenada (CoordConv, Liu et al. 2018): sin esto la
        # unica forma de que la conv "sepa" DONDE esta algo es a base de
        # receptive field, y el AdaptiveAvgPool2d de abajo lo tira todo de
        # golpe -- confirmado en el primer intento: el modelo colapsaba a
        # ~2 poses para frames de arenas completamente distintas (centro de
        # camara casi identico, focal con <3% de variacion). Con las
        # coordenadas explicitas en la entrada, cada canal de la conv puede
        # codificar posicion sin depender solo del campo receptivo.
        in_ch_full = in_ch + 2

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch_full, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            block(32, 64), block(64, 128), block(128, 256),
        )
        # pool a una rejilla pequeña (NO a 1x1): un vector global no distingue
        # "la linea azul esta a la izquierda" de "esta a la derecha" si la
        # activacion media por canal es parecida -- exactamente el colapso
        # que se midio. 6x4 conserva la disposicion gruesa sin disparar el
        # tamaño de la FC.
        self.pool = nn.AdaptiveAvgPool2d((4, 6))
        feat_dim = 256 * 4 * 6
        # coverage: fraccion de pixeles de cada clase presentes -- señal barata
        # y directa de "cuantas correspondencias hay", complementa a la conv.
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + in_ch, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )
        self.out_rot = nn.Linear(128, 6)
        self.out_center = nn.Linear(128, 3)
        self.out_focal = nn.Linear(128, 1)

        # inicializacion de salida: arranca cerca de una pose "tipica" de
        # retransmision (camara alta, a un lado de la pista, mirando al
        # centro) en vez de identidad -- converge mucho mas rapido que
        # arrancar de una pose aleatoria/degenerada.
        self.out_rot.weight.data.zero_()
        self.out_rot.bias.data.copy_(torch.tensor([1., 0., 0., 0., 0., -1.]))
        self.out_center.weight.data.zero_()
        self.out_center.bias.data.copy_(torch.tensor([30., -25., 18.]))
        self.out_focal.weight.data.zero_()
        self.out_focal.bias.data.fill_(2200.0)

    def forward(self, probs):
        B, _, h, w = probs.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=probs.device),
            torch.linspace(-1, 1, w, device=probs.device), indexing="ij")
        coord = torch.stack([xx, yy]).unsqueeze(0).expand(B, -1, -1, -1)
        coverage = probs.mean(dim=(2, 3))                 # (B, in_ch)
        z = self.pool(self.conv(torch.cat([probs, coord], dim=1))).flatten(1)
        z = self.mlp(torch.cat([z, coverage], dim=1))
        R = rot6d_to_matrix(self.out_rot(z))
        C = self.out_center(z)
        f = Fnn.softplus(self.out_focal(z)).squeeze(-1) + 100.0   # (B,) focal siempre positiva
        return R, C, f


# ------------------------------------------------------------------ perdida

def project_batch(H, pts_world):
    """H: (B,3,3).  pts_world: (P,2) metros.  -> (B,P,2) pixeles."""
    P = pts_world.shape[0]
    ones = torch.ones(P, 1, device=pts_world.device, dtype=pts_world.dtype)
    pts_h = torch.cat([pts_world, ones], dim=1)                     # (P,3)
    proj = torch.einsum("bij,pj->bpi", H, pts_h)                    # (B,P,3)
    z = proj[:, :, 2:3]
    z = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
    return proj[:, :, :2] / z


def reprojection_loss(H_pred, H_true, pts_world, sig, delta_px=50.0, pad=1920.0):
    """Huber sobre el error de reproyeccion en px a 1920 (sig = w/1920 por
    muestra, para que el delta signifique lo mismo en cualquier resolucion).

    Recorta pred Y verdad (los dos, no solo la resta) a una caja ampliada
    (1 ancho de frame de margen a cada lado) antes de medir distancia. Bug de
    diseño encontrado tras dos entrenamientos completos estancados en
    p50~850px: de los 56 puntos del template, muchos caen MUY fuera de
    cuadro segun el encuadre (visto: hasta -66000px en un solo frame, 19/56
    fuera de cuadro) -- exigir precision en PIXELES sobre esa posicion es un
    objetivo mal condicionado (un error angular minimo la mueve miles de px),
    y aunque Huber acota el GRADIENTE por punto, con tantos puntos asi de
    inestables el entrenamiento nunca converge fino. Al recortar los dos
    lados a la misma caja, un punto cuya verdad ya cae fuera del margen deja
    de pedir precision (basta con que la prediccion tambien caiga fuera, en
    la misma zona) -- degradacion suave en vez de objetivo imposible.
    """
    xy_pred = project_batch(H_pred, pts_world) / sig.view(-1, 1, 1)
    with torch.no_grad():
        xy_true = project_batch(H_true, pts_world) / sig.view(-1, 1, 1)
    lo = torch.tensor([-pad, -pad * 9 / 16], device=xy_pred.device)
    hi = torch.tensor([REF_WIDTH + pad, REF_WIDTH * 9 / 16 + pad * 9 / 16], device=xy_pred.device)
    xy_pred_c = torch.clamp(xy_pred, lo, hi)
    xy_true_c = torch.clamp(xy_true, lo, hi)
    dist = torch.linalg.norm(xy_pred_c - xy_true_c, dim=2)          # (B,P) px a 1920
    loss = Fnn.huber_loss(dist, torch.zeros_like(dist), delta=delta_px, reduction="none")
    return loss.mean(), dist


def reprojection_error_inframe(H_pred, H_true, pts_world, sig, w, h):
    """Error real (sin recortar) SOLO sobre los puntos que la H de verdad
    coloca dentro del frame -- para comparar con seg_to_homography.py, que
    mide igual (geom_error: solo puntos del template en cuadro)."""
    xy_pred = project_batch(H_pred, pts_world)
    with torch.no_grad():
        xy_true = project_batch(H_true, pts_world)
    inb = ((xy_true[..., 0] > 0) & (xy_true[..., 0] < w.view(-1, 1)) &
           (xy_true[..., 1] > 0) & (xy_true[..., 1] < h.view(-1, 1)))
    err = torch.linalg.norm(xy_pred - xy_true, dim=2) / sig.view(-1, 1)
    return err[inb]


# ------------------------------------------------------------------ datos

class SegProbDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        probs = torch.from_numpy(d["probs"].astype(np.float32))
        H = torch.from_numpy(d["H"].astype(np.float32))
        w, h = d["wh"].astype(np.float32)
        return probs, H, w, h


def list_cached(split):
    files = []
    for ds, _ in JOBS:
        files += sorted((ROOT / "datasets" / ds / "seg_probs" / split).glob("*.npz"))
    return files


# ------------------------------------------------------------------ entrenamiento

def evaluar(model, loader, pts_world, device):
    model.eval()
    errores, errores_cuadro = [], []
    with torch.no_grad():
        for probs, H_true, w, h in loader:
            probs, H_true = probs.to(device), H_true.to(device)
            w, h = w.to(device), h.to(device)
            R, C, f = model(probs)
            H_pred = pose_to_H(R, C, f * w / REF_WIDTH, w, h)
            sig = w / REF_WIDTH
            _, dist = reprojection_loss(H_pred, H_true, pts_world, sig)
            errores.append(dist.cpu().numpy().ravel())
            err_ib = reprojection_error_inframe(H_pred, H_true, pts_world, sig, w, h)
            errores_cuadro.append(err_ib.cpu().numpy().ravel())
    model.train()
    errores = np.concatenate(errores)
    errores_cuadro = np.concatenate(errores_cuadro) if errores_cuadro else np.array([])
    return errores, errores_cuadro


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_files = list_cached("train")
    val_files = list_cached("val")
    if not train_files:
        raise SystemExit("no hay cache -- corre antes training/cache_seg_probs.py")
    print("train {} imagenes   val {} imagenes   device={}".format(
        len(train_files), len(val_files), device))

    tpl_iihf = rink.build_template(rink.RINK_IIHF)
    tpl_nhl = rink.build_template(rink.RINK_NHL)
    assert np.allclose(tpl_iihf.shape, tpl_nhl.shape)
    # los 56 puntos no coinciden exactamente en metros entre IIHF/NHL (la pista
    # NHL es mas estrecha) -- se usa el promedio como objetivo de supervision,
    # la diferencia (~4m) es pequeña comparada con el resto de la geometria y
    # evita tener que condicionar el modelo por reglamento.
    pts_world = torch.from_numpy(((tpl_iihf + tpl_nhl) / 2).astype(np.float32)).to(device)

    train_loader = DataLoader(SegProbDataset(train_files), batch_size=args.batch,
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(SegProbDataset(val_files), batch_size=args.batch, num_workers=2)

    model = PoseHead().to(device)
    if args.init and Path(args.init).exists():
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"])
        print("pesos de partida:", args.init)
        # LIMITACION CONOCIDA: solo se restauran los pesos, no el estado del
        # optimizador/scheduler -- --init reinicia el coseno de LR desde el
        # pico, lo que da un salto brusco las primeras epocas (visto: p50 de
        # val empeora 4-5x antes de recuperarse). No afecta a un run limpio.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = ROOT / "runs" / "homography_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    mejor = np.inf

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for probs, H_true, w, h in train_loader:
            probs, H_true = probs.to(device), H_true.to(device)
            w, h = w.to(device), h.to(device)
            R, C, f = model(probs)
            H_pred = pose_to_H(R, C, f * w / REF_WIDTH, w, h)
            sig = w / REF_WIDTH
            loss, _ = reprojection_loss(H_pred, H_true, pts_world, sig)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        err_val, err_cuadro = evaluar(model, val_loader, pts_world, device)
        p50, p90 = np.median(err_val), np.percentile(err_val, 90)
        # p50/p90 "en cuadro" son los comparables directos con seg_to_homography.py
        # (que solo mide sobre puntos del template dentro del frame); los de
        # arriba (todos los puntos, recortados a la caja ampliada) son el que
        # de verdad optimiza la perdida, se imprimen como diagnostico de si
        # sigue bajando.
        if len(err_cuadro):
            p50c, p90c = np.median(err_cuadro), np.percentile(err_cuadro, 90)
        else:
            p50c = p90c = np.nan
        print("ep {:3d}  loss={:.2f}  val(recortado): p50={:6.1f}px p90={:7.1f}px  |  "
              "val(en cuadro): p50={:6.1f}px p90={:7.1f}px <25px={:4.0f}%  ({:.0f}s)".format(
            ep, np.mean(losses), p50, p90, p50c, p90c,
            100 * np.mean(err_cuadro < 25) if len(err_cuadro) else 0.0, time.time() - t0))

        torch.save({"model": model.state_dict(), "epoch": ep}, out_dir / "last.pt")
        if p50c < mejor:
            mejor = p50c
            torch.save({"model": model.state_dict(), "epoch": ep, "p50": p50c}, out_dir / "best.pt")

    print("\nmejor p50 en cuadro de val: {:.1f}px  ->  {}".format(mejor, out_dir / "best.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default=None, help="checkpoint de partida (para continuar un run)")
    args = ap.parse_args()
    run(args)
