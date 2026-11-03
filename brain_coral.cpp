/*
source venv/bin/activate
g++ -O3 -std=c++20 brain_coral.cpp -o brain_coral
*/

/*
pouzite knihovny potrebne pro vypocet a vizualizaci
*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

/*
definice vstupnich parametru, konfiguracni soubor pro ulohu
*/

struct Params
{
    int N = 300;  //nastaveni rozmeru ctvercove site

    int steps = 12000; //pocet kroku simulace
    int frames = 120;  //pocet snimku pro animaci

    double F = 0.030;  //tzv. feed rate, predstavuje prisun latky U do systemu, vetsi cislo = agresivnejsi rust
    double k = 0.055;  //tzv. kill rate, predstavuje miru zaniku latky V, male cislo = vice aktivni latky

    double dt = 1.0;   //delka casoveho kroku

    double D1 = 0.18;  //D1 az D4 predstavuji postupne difuzivitu ve vsech rozich site, jedna se o primarni nastaveni pro spusteni 
    double D2 = 0.12;
    double D3 = 0.08;
    double D4 = 0.04;
};

/*
trida BrainCoral predstavuje 1 realizaci vypoctu Gray-Scottova modelu difuzniho rustu, obsahuje potrebne parametry a fce pro model
*/

class BrainCoral
{
private:

    Params p;   //obsahuje parametry popsane v struct Params, odsud si algoritmus parametry pro model bere

    std::vector<double> U,V;    //pole koncentraci latek U a V, latka U = zdrojova latka, latka V = koral
    std::vector<double> Un,Vn;  //nove hodnoty koncentraci latek U a V

    int idx(int x,int y) const  //prevadi souradnice (x,y) na index jednorozmerneho pole
    {
        return y*p.N+x;
    }

    double localDu(int x,int y) const   //resi lokalni difuzivitu
    {
        double rx =
            static_cast<double>(x)/(p.N-1); //normalizace souradnic x

        double ry =
            static_cast<double>(y)/(p.N-1); //normalizace souradnic y

        double w1=(1-rx)*(1-ry);
        double w2=rx*(1-ry);
        double w3=(1-rx)*ry;
        double w4=rx*ry;        //w1 az w4 jsou vahy bylinearni interpolace

        return
            w1*p.D1+
            w2*p.D2+
            w3*p.D3+
            w4*p.D4;
    }

/*
realizace pocatecni podminky, konstruktor pro BrainCoral
*/

public:

    explicit BrainCoral(const Params& params)   //prevzeti parametru Params, uzivatelske zadani parametru D1 az D4
    : p(params)
    {
        U.assign(p.N*p.N,1.0);  //nastaveni pocatecni koncentraci latky U, U(0)=1
        V.assign(p.N*p.N,0.0);  //nastaveni pocatecni koncentrace latky V, V(0)=0

        Un=U;   //inicializace
        Vn=V;

        int c=p.N/2;    //vypocet stredu domeny = odtud zacne koral rust

        std::mt19937 rng(42);   //generator pseudonahodnych cisel, stejny seed rng

        std::uniform_real_distribution<double>
        noise(-0.01,0.01);      //generace sumu v intervalu viz cisla, slouzi k naruseni pocatecni podminky -> ziskavame nesymetricke reseni

        for(int y=c-15;y<=c+15;y++)     //prochazka stredovou oblasti pro naruseni (generaci) prostredi = vytvoreni bodove poruchy pro inicializaci
        {
            for(int x=c-15;x<=c+15;x++)
            {
                int i=idx(x,y);         //prevod souradnic (x,y) na index

                U[i]=0.5+noise(rng);    //vlastni pocatecni podminka pro U
                V[i]=0.25+noise(rng);   //vlastni pocatecni podminka pro V
            }
        }
    }

/*
ukladani vysledku (vizualizace)
*/

    void saveFrame(int frame)   //ulozi stav koncentrace latky V
    {
        std::filesystem::create_directories("data");    //vytvoreni adresare

        char filename[128];     //vymezeni prostoru v pameti

        sprintf(        //vytvoreni souboru data_cislo, cisla jsou s nulami pro spravne razeni za sebe
            filename,
            "data/frame_%04d.dat",
            frame
        );

        std::ofstream out(      //otevreni vystupniho souboru, soubor je binarni (rychlejsi nez zapisovat jako text)
            filename,
            std::ios::binary
        );

        out.write(      //vlastni zapis dat, ulozeni dat V do vystupniho binarniho souboru
            reinterpret_cast<char*>(V.data()),  //reinterpret pro fci write, ta ocekava char*, V.data je ale double -> potreba prevodu
            V.size()*sizeof(double)
        );
    }


/*
vlastni numericke reseni Gray-Scottova modelu, reseno explicitinim Eulerem, postupne vypocte v kazdem bode:
Laplaceuv operator, reakcni clen, Euleruv krok, nove hodnoty Un a Vn
*/


    void step()
    {
        for(int y=1;y<p.N-1;y++)        //postupny pruchod vnitrnimi body site NxN, neaktualizuje hranicni body
        {
            for(int x=1;x<p.N-1;x++)
            {
                int i=idx(x,y);

                double u=U[i];  //nacteni koncentrace latky U v bode
                double v=V[i];  //nacteni koncentrace latky V v bode

                double Lu=      //Laplaceuv operator pro U, je diskretni, modeluje difuzi
                    U[idx(x-1,y)]
                  + U[idx(x+1,y)]
                  + U[idx(x,y-1)]
                  + U[idx(x,y+1)]
                  -4.0*u;

                double Lv=      //Laplaceuv operator pro V, je diskretni, modeluje difuzi
                    V[idx(x-1,y)]
                  + V[idx(x+1,y)]
                  + V[idx(x,y-1)]
                  + V[idx(x,y+1)]
                  -4.0*v;

                double uvv=u*v*v;   //reakcni nelinearni difuzni clen, vytvari vzory

                double Du=localDu(x,y); //nacteni lokalni difuzivity U v bode
                double Dv=0.5*Du;   //urceni difuzivity latky V, zde pro nasi realizaci je pro jednoduchost Dv vzato jako polovina Du (mozno zmenit)

                Un[i]=std::clamp(      //Euleruv krok reseni, clamp omezuje hodnoty na interval <0,1>
                    u+p.dt*(
                    Du*Lu
                    -uvv
                    +p.F*(1-u)),
                    0.0,
                    1.0
                );

                Vn[i]=std::clamp(      //Euleruv krok reseni, clamp omezuje hodnoty na interval <0,1>
                    v+p.dt*(
                    Dv*Lv
                    +uvv
                    -(p.F+p.k)*v),
                    0.0,
                    1.0
                );
            }
        }

        U.swap(Un);     //prohozeni pole U pro vypocet dalsiho stavu
        V.swap(Vn);     //prohozeni pole V pro vypocet dalsiho stavu
    }

/*
generace python skriptu, ktery postupne projde:
nacteni vygenerovanych snimku, prevede numericka data ze snimku na obrazky, vytvori pseudorelief pomoci nasviceni (LightSource), sestavi animaci a ulozi ji jako gif;
jedna se tak o sestaveni vizualizace, nejde jiz o vypocty
*/

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

/*
void fce pro rizeni celeho kodu, postupne projde numericke reseni a pote na jeho zaklade s pomoci python3 vygeneruje vizualizaci
*/

    void run()
    {
        std::cout << "\nRunning simulation:\n";

        std::cout       //vypsani jednotlivych difuzivit
            << "D1 = " << p.D1 << "\n"
            << "D2 = " << p.D2 << "\n"
            << "D3 = " << p.D3 << "\n"
            << "D4 = " << p.D4 << "\n\n";

        saveFrame(0);   //ulozeni pocatecniho stavu -> zacatek "zacina" spolu s pocatecni podminkou

        int frame=1;    //pocitadlo snimku

        for(int t=1;t<=p.steps;t++)     //casova smycka, resi casovy krok pro Eulera
        {
            step();     //numericke reseni Eulerovou explicitni metodou

            int curr =
                t*p.frames/p.steps;     //rozhoduje zda "nastal" cas ulozit snimek = neuklada vsechny, selektuje

            int prev =
                (t-1)*p.frames/p.steps;

            if(curr!=prev)      //ulozeni snimku
            {
                saveFrame(frame);

                std::cout
                    << "saved frame "       //informacni vypis, vypisuje postup simulace polemi
                    << frame
                    << "/"
                    << p.frames
                    << "\n";

                ++frame;
            }
        }

        std::cout
            << "\nGenerating visualization...\n";   //generace vizualizace

        generatePythonScript();

        [[maybe_unused]]
        int status =
            std::system("python3 visualize.py");    //spusteni python3 z prostredi C++
    }
};

int main(int argc,char* argv[])
{
    Params p;

    if(argc == 5)       //moznost pro zadani difuzivit D1 az D4 pomoci prikazoveho radku (poradi D1 D2 D3 D4)
    {
        p.D1 = std::stod(argv[1]);      //prevod textu na cislo pro jednotlive zadavane difuzivity D
        p.D2 = std::stod(argv[2]);
        p.D3 = std::stod(argv[3]);
        p.D4 = std::stod(argv[4]);
    }
    else                //neni-li zadana difuzivita D1 az D4 pouzije primarni nastaveni
    {
        std::cout       //vypis napovedy
        << "\nUsage:\n"
        << "./brain_coral D1 D2 D3 D4\n\n"
        << "Example:\n"
        << "./brain_coral 0.18 0.12 0.08 0.04\n\n"
        << "Using default values.\n\n";
    }

    BrainCoral coral(p);

    coral.run();        //spusteni simulace

    return 0;
}
