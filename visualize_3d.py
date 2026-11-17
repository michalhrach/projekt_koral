"""vizualizace 3d dat z programu brain_coral_3d.cpp"""




from pathlib import Path                 # prace se slozkami a soubory

import numpy as np                        # nacteni a zpracovani objemovych dat
import matplotlib                        # vykreslovaci knihovna

matplotlib.use('Agg')                     # vykreslovani bez grafickeho okna

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes




# slozka s vypoctenymi snimky a konfiguraci
base = Path('frames3d')                   # vystupni slozka C++ programu

N, frames, iso = base.joinpath('config.txt').read_text().split()

N = int(N)                                # rozmer prostorove site
frames = int(frames)                      # pozadovany pocet snimku
iso = float(iso)                          # hodnota koncentrace pro izoplochu

files = sorted(base.glob('frame_*.bin'))  # serazeni snimku podle nazvu

if len(files) != frames:
    raise RuntimeError(
        f'Ocekavano {frames} snimku, nalezeno {len(files)}'
    )




out = Path('vysledky3d')
out.mkdir(exist_ok=True)

# priprava okna pro 3d vykresleni
fig = plt.figure(
    figsize=(6, 6),
    dpi=95,
    facecolor='#101821'
)

ax = fig.add_subplot(
        111,
    projection='3d',
    computed_zorder=False
)

fig.subplots_adjust(
    left=0.02,
    right=0.98,
    bottom=0.02,
    top=0.96
)

writer = PillowWriter(fps=8)               # 8 snimku za sekundu GIFu




# postupne nacteni snimku a zapis animace
with writer.saving(
    fig,
    str(out / 'brain_coral_3d.gif'),
    dpi=95
):
    for j, f in enumerate(files):
        # nacteni hodnot koncentrace v z binarniho souboru.
        v = np.fromfile(
            f,
            dtype=np.float32
        )

        if v.size != N ** 3:
            raise RuntimeError(f'Neplatny snimek {f}')

        # prevody z jednorozmerneho pole na prostorovou sit N x N x N.
        volume = v.reshape((N, N, N))
        # vymazani predchoziho obrazu a nastaveni 3d os.
        ax.cla()
        ax.set_facecolor('#101821')

        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_zlim(0, N)

        ax.set_box_aspect(
            (1, 1, 1),
            zoom=1.04
        )

        ax.set_axis_off()                   # skryti souradnicovych os

        # pootoceni kamery v kazdem dalsim snimku.
        ax.view_init(
            elev=23,
            azim=35 + 310 * j / max(1, frames - 1)
        )
        # vytvoreni izoplochy jen v pripade, ze je prah iso
        #     uvnitr rozsahu koncentraci v aktualnim snimku.
        if np.min(volume) < iso < np.max(volume):

            verts, faces, _, _ = marching_cubes(
                volume,
                iso,
                step_size=3,
                allow_degenerate=False
            )

            # trojuhelnikova sit povrchu.
            # step_size=3 snizuje mnozstvi trojuhelniku pri vizualizaci.
            tris = verts[faces]
        # vypocet normal trojuhelniku a jednoducheho osvetleni.
            normal = np.cross(
                tris[:, 1] - tris[:, 0],
                tris[:, 2] - tris[:, 0]
            )

            size = np.linalg.norm(
                normal,
                axis=1
            )

            normal /= np.maximum(size[:, None], 1e-10)

            # smer osvetleni; normalizace na jednotkovy vektor.
            lamp = np.array([0.35, -0.4, 0.85])
            lamp /= np.linalg.norm(lamp)

            brightness = np.clip(
                0.45 + 0.55 * np.abs(normal @ lamp),
                0.2,
                1.0
            )

            # zakladni barva povrchu, upravena podle osvetleni.
            basecolor = np.array([0.93, 0.66, 0.39])

            colors = np.column_stack([
                brightness[:, None] * basecolor,
                np.ones(len(brightness))
            ])
        # sestaveni 3d objektu z trojuhelnikove site.
                    mesh = Poly3DCollection(
                tris,
                facecolors=colors,
                edgecolors='none',
                linewidth=0,
                alpha=1,
                zsort='average'
            )

            ax.add_collection3d(mesh)
        # popisek a ulozeni aktualniho obrazku do gifu.
        ax.set_title(
            f'Gray–Scott 3D   |   snimek {j + 1}/{frames}',
            color='white',
            fontsize=12,
            pad=4
        )

        writer.grab_frame()

        print(
            f'3D GIF: {j + 1}/{frames}',
            flush=True
        )




# ukonceni vykreslovani
plt.close(fig)
print('Hotovo:', out / 'brain_coral_3d.gif')
