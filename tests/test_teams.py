"""Team from jersey colour (sportcal/core/teams.py): synthetic crops of two shirt colours and one of bare ice."""
import numpy as np

from sportcal.core.teams import fit_teams, jersey_histogram, team_of


def _crop(bgr, rng):
    """Histogram of a 40 x 100 box: white ice around, a noisy shirt colour on the torso band (rows 25-65)."""
    frame = np.full((100, 40, 3), 235, np.uint8)
    frame[25:65, 5:35] = np.clip(np.array(bgr) + rng.normal(0, 12, (40, 30, 3)), 0, 255).astype(np.uint8)
    return jersey_histogram(frame, (0, 0, 40, 100))


def test_two_shirt_colours_split_into_two_confident_teams_and_bare_ice_has_no_team():
    rng = np.random.default_rng(0)
    green = [_crop((40, 180, 40), rng) for _ in range(15)]
    blue = [_crop((200, 60, 30), rng) for _ in range(15)]
    teams, ratio = team_of(green + blue, fit_teams(green + blue))
    assert len(set(teams[:15])) == 1 and len(set(teams[15:])) == 1 and teams[0] != teams[15]
    assert ratio.max() < 0.8
    assert jersey_histogram(np.full((100, 40, 3), 235, np.uint8), (0, 0, 40, 100)) is None
