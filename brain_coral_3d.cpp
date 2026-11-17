/*
source venv/bin/activate
g++ -O3 -std=c++17 brain_coral_3d.cpp -o brain_coral_3d
./brain_coral_3d
*/

// knihovny pro vypocet, praci se soubory a spusteni vizualizace

#include <algorithm>    // std::clamp, std::max
#include <cmath>         // matematicke operace
#include <cstdio>       // std::snprintf pro sestaveni nazvu souboru
#include <cstdlib>      // std::system pro spusteni Pythonu a otevreni GIFu
#include <filesystem>   // vytvareni a mazani vystupnich slozek
#include <fstream>      // zapis binarnich dat a konfigurace
#include <iostream>     // vypisy informaci do terminalu
#include <random>       // pocatecni nahodne poruchy koncentraci
#include <stdexcept>    // vyjimky pri chybnym parametrech a zapisu
#include <string>       // zpracovani parametru zadanych v terminalu
#include <vector>       // pole hodnot koncentraci U a V


// vstupni parametry pro vypocet a zobrazeni simulace

struct Params
{
    int N = 72;             // rozmer prostorove site N x N x N; napr. 100 = 1 milion bodu

    int steps = 2200;       // celkovy pocet kroku explicitni Eulerovy metody
    int frames = 28;        // pocet ukladanych snimku vcetne pocatecniho stavu

    double F = 0.035;       // feed rate - prisun latky U do systemu
    double k = 0.060;       // kill rate - odsun / zanik latky V

    double dt = 0.5;        // delka jednoho casoveho kroku

    double Du = 0.18;       // difuzni koeficient latky U
    double Dv = 0.09;       // difuzni koeficient latky V

    double iso = 0.14;      // prah koncentrace V pro vykresleni 3D izoplochy

    bool visualize = true;  // po vypoctu automaticky spustit Python a vytvorit GIF
    bool openGif = true;    // zkusit GIF otevrit ve Windows pri praci ve WSL
};


// trida pro jednu realizaci 3d gray-scottova modelu

class BrainCoral
{
private:
    Params p;    // ulozene parametry simulace

    std::vector<float> U, V;     // koncentrace latek U a V v aktualnim casovem kroku
    std::vector<float> Un, Vn;   // nove hodnoty koncentraci pro nasledujici krok


    // prevod souradnic 3d site na index jednorozmerneho pole

    int idx(int x, int y, int z) const
    {
        return (z * p.N + y) * p.N + x;
    }


    // priprava pocatecni kulove poruchy koncentraci

    void seed()
    {
        std::mt19937 gen(42);  // generator pseudonahodnych cisel

        std::uniform_real_distribution<float> noise(
            -0.025f,
             0.025f
        );

        const int c = p.N / 2;                          // stred vypocetni oblasti
        const double r = std::max(5.0, p.N * 0.12);    // polomer pocatecni poruchy

        for (int z = 1; z < p.N - 1; ++z)
        {
            for (int y = 1; y < p.N - 1; ++y)
            {
                for (int x = 1; x < p.N - 1; ++x)
                {
                    const double dx = x - c;  // vzdalenost od stredu ve smeru x
                    const double dy = y - c;  // vzdalenost od stredu ve smeru y
                    const double dz = z - c;  // vzdalenost od stredu ve smeru z

                    // test, zda bod lezi uvnitr pocatecni koule.
                    if (dx * dx + dy * dy + dz * dz < r * r)
                    {
                        int i = idx(x, y, z);

                        U[i] = 0.5f + noise(gen);   // pocatecni koncentrace U
                        V[i] = 0.25f + noise(gen);  // pocatecni koncentrace V
                    }
                }
            }
        }
    }
    

    // ulozeni koncentrace v v aktualnim case do binarniho souboru

    void writeFrame(int n) const
    {
        std::filesystem::create_directories("frames3d"); // slozka pro vysledna data

        char name[100];  // vyhrazeni prostoru pro cestu k souboru

        std::snprintf(
            name,
            sizeof(name),
            "frames3d/frame_%04d.bin",
            n
        );  // napriklad frames3d/frame_0001.bin

        std::ofstream f(
            name,
            std::ios::binary
        );

        if (!f)
        {
            throw std::runtime_error("Nelze otevrit vystupni soubor");
        }

        f.write(
            reinterpret_cast<const char*>(V.data()),
            static_cast<std::streamsize>(V.size() * sizeof(float))
        );

        if (!f)
        {
            throw std::runtime_error("Chyba pri zapisu snimku");
        }
    }


    // numericke reseni 3d gray-scottova modelu explicitni eulerovou metodou

    void step()
    {
        const int N = p.N;      // pocet bodu v kazdem prostorovem smeru
        const int N2 = N * N;   // posun v poli o jednu vrstvu ve smeru z

        // prevedeni hodnot na float, protoze pole koncentraci jsou typu float.
        const float dt = static_cast<float>(p.dt);
        const float du = static_cast<float>(p.Du);
        const float dv = static_cast<float>(p.Dv);
        const float F = static_cast<float>(p.F);
        const float k = static_cast<float>(p.k);

        // prochazime pouze vnitrni body site, bez hranicnich rovin.
        for (int z = 1; z < N - 1; ++z)
        {
            for (int y = 1; y < N - 1; ++y)
            {
                for (int x = 1; x < N - 1; ++x)
                {
                    int i = idx(x, y, z);  // index aktualniho bodu

                    float u = U[i];        // aktualni koncentrace latky U
                    float v = V[i];        // aktualni koncentrace latky V

                    // laplaceuv operator pro U: 6 sousedu minus 6krat stred.
                    // pro h=1 neni nutne delit vysledek h^2.
                    float Lu =
                          U[i - 1]
                        + U[i + 1]
                        + U[i - N]
                        + U[i + N]
                        + U[i - N2]
                        + U[i + N2]
                        - 6.0f * u;

                    // laplaceuv operator pro V, analogicky jako u latky U.
                    float Lv =
                          V[i - 1]
                        + V[i + 1]
                        + V[i - N]
                        + V[i + N]
                        + V[i - N2]
                        + V[i + N2]
                        - 6.0f * v;

                    // nelinearni reakcni clen predstavujici U*V^2.
                    // v rovnici U se odcita, v rovnici V se pricita.
                    float reaction = u * v * v;

                    // euleruv krok: vypocet nove koncentrace U.
                    Un[i] = std::clamp(
                        u + dt * (
                            du * Lu
                            - reaction
                            + F * (1.0f - u)
                        ),
                        0.0f,
                        1.0f
                    );
                    
                    // euleruv krok: vypocet nove koncentrace V.
                    Vn[i] = std::clamp(
                        v + dt * (
                            dv * Lv
                            + reaction
                            - (F + k) * v
                        ),
                        0.0f,
                        1.0f
                    );
                }
            }
        }

        // nove vypoctene koncentrace se stanou vychozimi pro dalsi krok.
        U.swap(Un);
        V.swap(Vn);
    }


public:

    // konstruktor: kontrola parametru a nastaveni pocatecnich podminek

    explicit BrainCoral(Params params)
        : p(params)
    {
        // zakladni kontrola velikosti site a poctu snimku.
        if (p.N < 24 || p.steps < 1 || p.frames < 2 || p.frames > p.steps + 1)
        {
            throw std::runtime_error(
                "Neplatne parametry site nebo poctu kroku/snimku"
            );
        }

        // kontrola hodnot reakcnich a difuznich parametru.
        if (p.dt <= 0 || p.Du <= 0 || p.Dv <= 0 ||
            p.F < 0 || p.k < 0 || p.iso <= 0)
        {
            throw std::runtime_error("Neplatne fyzikalni parametry");
        }

        // podminka pro explicitni Eulerovu metodu a samotnou 3D difuzi
        // pri h = 1: dt * max(Du,Dv) <= 1/6.
        if (p.dt * std::max(p.Du, p.Dv) > 1.0 / 6.0)
        {
            throw std::runtime_error(
                "Prilis velke dt: pro samotnou 3D difuzi plati dt*Dmax <= 1/6 (h=1)"
            );
                    }

        // celkovy pocet bodu prostorove site (N*N*N).
        const size_t count = static_cast<size_t>(p.N) * p.N * p.N;

        U.assign(count, 1.0f);  // na pocatku U = 1 v cele oblasti
        V.assign(count, 0.0f);  // na pocatku V = 0 v cele oblasti

        Un = U;                  // kopie poli pro novy casovy krok
        Vn = V;

        seed();                  // vytvoreni pocatecni poruchy ve stredu domeny
    }


    // spusteni vypoctu, ukladani snimku a vytvoreni gifu

    void run()
    {
        // odstraneni starych snimku, aby se nemichaly s novym vypoctem.
        std::filesystem::remove_all("frames3d");
        std::filesystem::create_directories("frames3d");

        // metadata pro Python: velikost site, pocet snimku a prah izoplochy.
        std::ofstream config("frames3d/config.txt");
        config << p.N << " " << p.frames << " " << p.iso << "\n";
        config.close();

        writeFrame(0);  // ulozeni pocatecniho stavu v case t=0

        int saved = 1;  // pocet dosud ulozenych snimku, vcetne pocatecniho

        // hlavni casova smycka numerickeho vypoctu.
        for (int t = 1; t <= p.steps; ++t)
        {
            step();  // provedeni jednoho casoveho kroku explicitniho Eulera

            // snímky rozmistime priblizne rovnomerne po celou dobu simulace.
            // prvni snimek (t=0) uz byl ulozen pred smyckou.
            if (saved < p.frames &&
                t == (saved * p.steps) / (p.frames - 1))
            {
                writeFrame(saved);

                std::cout
                    << "Snimek " << saved + 1
                    << "/" << p.frames
                    << " (krok " << t << ")\n";

                ++saved;
                           }
        }

        // pojistka pro zaokrouhlovani pri rozdeleni kroku na snimky.
        if (saved < p.frames)
        {
            writeFrame(saved);
            ++saved;
        }

        if (saved != p.frames)
        {
            throw std::runtime_error("Nesouhlasi pocet vystupnich snimku");
        }

        std::cout << "Hotovo. 3D data jsou ve frames3d/.\n";

        // automaticke spusteni vizualizace pomoci samostatneho Python skriptu.
        if (p.visualize)
        {
            const int result = std::system("python3 visualize_3d.py");

            if (result != 0)
            {
                std::cerr
                    << "Nepodarilo se spustit vizualizaci; "
                    << "zkus python3 -m pip install "
                    << "numpy matplotlib scikit-image pillow\n";
                return;
            }

            // pokus o otevreni hotoveho GIFu ve Windows pri pouziti WSL.
            if (p.openGif)
            {
                const int rc = std::system(
                    "command -v explorer.exe >/dev/null 2>&1 && "
                    "explorer.exe vysledky3d/brain_coral_3d.gif >/dev/null 2>&1"
                );

                if (rc != 0)
                {
                    std::cout
                        << "GIF: vysledky3d/brain_coral_3d.gif "
                        << "(otevri rucne)\n";
                }
            }
        }
    }
};


// hlavni funkce: nacteni argumentu a spusteni simulace

int main(int argc, char** argv)
{
    Params p;  // pocatecni nastaveni podle struktury Params

    try
    {
        // postupne prochazime vsechny argumenty z prikazove radky.
        for (int a = 1; a < argc; ++a)
        {
            const std::string key = argv[a];

            if (key == "--no-gif")
            {
                p.visualize = false;
                continue;
            }

            if (key == "--no-open")
            {
                p.openGif = false;
                continue;
            }

            // vypis dostupnych parametru a ukonceni programu.
            if (key == "--help")
            {
                std::cout
                    << "Pouziti: ./brain_coral_3d "
                    << "[--N 72] [--steps 2200] [--frames 28] "
                    << "[--F .035] [--k .060] [--dt .5] "
                    << "[--Du .18] [--Dv .09] [--iso .14] "
                    << "[--no-gif] [--no-open]\n";

                return 0;
            }

            // u ostatnich parametru musi nasledovat ciselna hodnota.
            if (a + 1 >= argc)
            {
                throw std::runtime_error("Chybi hodnota argumentu " + key);
            }

            const std::string value = argv[++a];

            // prevod textovych argumentu na cisla a prirazeni do Params.
            if (key == "--N")
            {
                                p.N = std::stoi(value);
            }
            else if (key == "--steps")
            {
                p.steps = std::stoi(value);
            }
            else if (key == "--frames")
            {
                p.frames = std::stoi(value);
            }
            else if (key == "--F")
            {
                p.F = std::stod(value);
            }
            else if (key == "--k")
            {
                p.k = std::stod(value);
            }
            else if (key == "--dt")
            {
                p.dt = std::stod(value);
            }
            else if (key == "--Du")
            {
                p.Du = std::stod(value);
            }
            else if (key == "--Dv")
            {
                p.Dv = std::stod(value);
            }
            else if (key == "--iso")
            {
                p.iso = std::stod(value);
            }
            else
            {
                throw std::runtime_error("Neznamy argument: " + key);
            }
        }

        BrainCoral coral(p);  // vytvoreni modelu a inicializace koncentraci
        coral.run();          // spusteni simulace a pripadne vizualizace
    }
    catch (const std::exception& e)
    {
        std::cerr << "Chyba: " << e.what() << "\n";
        return 1;
    }
}
