#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

struct Params
{
    int N = 300;

    int steps = 12000;
    int frames = 120;

    double F = 0.030;
    double k = 0.055;

    double dt = 1.0;

    double D1 = 0.18;
    double D2 = 0.12;
    double D3 = 0.08;
    double D4 = 0.04;
};

class BrainCoral
{
private:

    Params p;

    std::vector<double> U,V;
    std::vector<double> Un,Vn;

    int idx(int x,int y) const
    {
        return y*p.N+x;
    }

    double localDu(int x,int y) const
    {
        double rx =
            static_cast<double>(x)/(p.N-1);

        double ry =
            static_cast<double>(y)/(p.N-1);

        double w1=(1-rx)*(1-ry);
        double w2=rx*(1-ry);
        double w3=(1-rx)*ry;
        double w4=rx*ry;

        return
            w1*p.D1+
            w2*p.D2+
            w3*p.D3+
            w4*p.D4;
    }

public:

    explicit BrainCoral(const Params& params)
    : p(params)
    {
        U.assign(p.N*p.N,1.0);
        V.assign(p.N*p.N,0.0);

        Un=U;
        Vn=V;

        int c=p.N/2;

        std::mt19937 rng(42);

        std::uniform_real_distribution<double>
        noise(-0.01,0.01);

        for(int y=c-15;y<=c+15;y++)
        {
            for(int x=c-15;x<=c+15;x++)
            {
                int i=idx(x,y);

                U[i]=0.5+noise(rng);
                V[i]=0.25+noise(rng);
            }
        }
    }

    void saveFrame(int frame)
    {
        std::filesystem::create_directories("data");

        char filename[128];

        sprintf(
            filename,
            "data/frame_%04d.dat",
            frame
        );

        std::ofstream out(
            filename,
            std::ios::binary
        );

        out.write(
            reinterpret_cast<char*>(V.data()),
            V.size()*sizeof(double)
        );
    }

    void step()
    {
        for(int y=1;y<p.N-1;y++)
        {
            for(int x=1;x<p.N-1;x++)
            {
                int i=idx(x,y);

                double u=U[i];
                double v=V[i];

                double Lu=
                    U[idx(x-1,y)]
                  + U[idx(x+1,y)]
                  + U[idx(x,y-1)]
                  + U[idx(x,y+1)]
                  -4.0*u;

                double Lv=
                    V[idx(x-1,y)]
                  + V[idx(x+1,y)]
                  + V[idx(x,y-1)]
                  + V[idx(x,y+1)]
                  -4.0*v;

                double uvv=u*v*v;

                double Du=localDu(x,y);
                double Dv=0.5*Du;

                Un[i]=std::clamp(
                    u+p.dt*(
                    Du*Lu
                    -uvv
                    +p.F*(1-u)),
                    0.0,
                    1.0
                );

                Vn[i]=std::clamp(
                    v+p.dt*(
                    Dv*Lv
                    +uvv
                    -(p.F+p.k)*v),
                    0.0,
                    1.0
                );
            }
        }

        U.swap(Un);
        V.swap(Vn);
    }



    void generatePythonScript()
{
    std::ofstream py("visualize.py");

    py << R"(

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from matplotlib.colors import LightSource
from matplotlib.colors import LinearSegmentedColormap

N = 300

files = sorted(
    Path("data").glob("*.dat")
)

# Korálová barevná mapa
coral = LinearSegmentedColormap.from_list(
    "coral",
    [
        "#f5f1e8",
        "#d9c3a5",
        "#c69c72",
        "#8b5a3c",
        "#4a2d21"
    ]
)

fig, ax = plt.subplots(
    figsize=(8,8),
    dpi=150
)

ax.axis("off")

frames = []

light = LightSource(
    azdeg=315,
    altdeg=45
)

for f in files:

    arr = np.fromfile(
        f,
        dtype=np.float64
    ).reshape(N,N)

    # zesílení reliéfu
    arr = arr * 10.0

    rgb = light.shade(
        arr,
        cmap=coral,
        vert_exag=25,
        blend_mode='soft'
    )

    img = ax.imshow(
        rgb,
        animated=True
    )

    frames.append([img])

ani = animation.ArtistAnimation(
    fig,
    frames,
    interval=80,
    blit=True
)

ani.save(
    "brain_coral_relief.gif",
    writer="pillow"
)

try:
    ani.save(
        "brain_coral_relief.mp4",
        writer="ffmpeg"
    )
except:
    print("MP4 export skipped.")

print("Relief animation created.")

)";
}


    void run()
    {
        std::cout << "\nRunning simulation:\n";

        std::cout
            << "D1 = " << p.D1 << "\n"
            << "D2 = " << p.D2 << "\n"
            << "D3 = " << p.D3 << "\n"
            << "D4 = " << p.D4 << "\n\n";

        saveFrame(0);

        int frame=1;

        for(int t=1;t<=p.steps;t++)
        {
            step();

            int curr =
                t*p.frames/p.steps;

            int prev =
                (t-1)*p.frames/p.steps;

            if(curr!=prev)
            {
                saveFrame(frame);

                std::cout
                    << "saved frame "
                    << frame
                    << "/"
                    << p.frames
                    << "\n";

                ++frame;
            }
        }

        std::cout
            << "\nGenerating visualization...\n";

        generatePythonScript();

        [[maybe_unused]]
        int status =
            std::system("python3 visualize.py");
    }
};

int main(int argc,char* argv[])
{
    Params p;

    if(argc == 5)
    {
        p.D1 = std::stod(argv[1]);
        p.D2 = std::stod(argv[2]);
        p.D3 = std::stod(argv[3]);
        p.D4 = std::stod(argv[4]);
    }
    else
    {
        std::cout
        << "\nUsage:\n"
        << "./brain_coral D1 D2 D3 D4\n\n"
        << "Example:\n"
        << "./brain_coral 0.18 0.12 0.08 0.04\n\n"
        << "Using default values.\n\n";
    }

    BrainCoral coral(p);

    coral.run();

    return 0;
}