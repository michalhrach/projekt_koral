// Gray-Scott 3D — navazuje na puvodni BrainCoral 2D.
// Kompilace: g++ -O3 -std=c++17 brain_coral_3d.cpp -o brain_coral_3d
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Params {
    int N = 72;             // N x N x N voxelu (pro 100^3 zadej --N 100)
    int steps = 2200;       // pocet kroku Eulerovy metody
    int frames = 28;        // pocet ulozenych stavu (vcetne pocatku)
    double F = 0.035;       // prisun U
    double k = 0.060;       // odsun/zanik V
    double dt = 0.5;       // casovy krok
    double Du = 0.18;      // difuze U
    double Dv = 0.09;      // difuze V
    double iso = 0.14;     // vykreslovaci prah koncentrace V
    bool visualize = true; // automaticky vyrobit GIF
    bool openGif = true;   // ve WSL zkusit otevrit GIF ve Windows
};

class BrainCoral {
    Params p;
    std::vector<float> U, V, Un, Vn;
    int idx(int x, int y, int z) const { return (z*p.N + y)*p.N + x; }

    void seed() {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> noise(-0.025f,0.025f);
        const int c=p.N/2;
        const double r=std::max(5.,p.N*0.12);
        for (int z=1;z<p.N-1;++z)
            for(int y=1;y<p.N-1;++y)
                for(int x=1;x<p.N-1;++x) {
                    const double dx=x-c,dy=y-c,dz=z-c;
                    if(dx*dx+dy*dy+dz*dz < r*r) {
                        int i=idx(x,y,z);
                        U[i]=0.5f+noise(gen);
                        V[i]=0.25f+noise(gen);
                    }
                }
    }

    void writeFrame(int n) const {
        std::filesystem::create_directories("frames3d");
        char name[100]; std::snprintf(name,sizeof(name),"frames3d/frame_%04d.bin",n);
        std::ofstream f(name,std::ios::binary);
        if(!f) throw std::runtime_error("Nelze otevrit vystupni soubor");
        f.write(reinterpret_cast<const char*>(V.data()),
                static_cast<std::streamsize>(V.size()*sizeof(float)));
        if(!f) throw std::runtime_error("Chyba pri zapisu snimku");
    }

    void step() {
        const int N=p.N, N2=N*N;
        const float dt=static_cast<float>(p.dt);
        const float du=static_cast<float>(p.Du),dv=static_cast<float>(p.Dv);
        const float F=static_cast<float>(p.F),k=static_cast<float>(p.k);
        // Okraje drzi U=1, V=0 (Dirichletovy podminky).
        for(int z=1;z<N-1;++z) for(int y=1;y<N-1;++y) for(int x=1;x<N-1;++x) {
            const int i=idx(x,y,z);
            const float u=U[i],v=V[i];
            // 7bodovy 3D Laplaceuv operator; prostorovy krok h=1.
            const float Lu=U[i-1]+U[i+1]+U[i-N]+U[i+N]+U[i-N2]+U[i+N2]-6.f*u;
            const float Lv=V[i-1]+V[i+1]+V[i-N]+V[i+N]+V[i-N2]+V[i+N2]-6.f*v;
            const float reaction=u*v*v;
            Un[i]=std::clamp(u+dt*(du*Lu-reaction+F*(1.f-u)),0.f,1.f);
            Vn[i]=std::clamp(v+dt*(dv*Lv+reaction-(F+k)*v),0.f,1.f);
        }
        U.swap(Un);V.swap(Vn);
    }

public:
    explicit BrainCoral(Params params):p(params) {
        if(p.N<24 || p.steps<1 || p.frames<2 || p.frames>p.steps+1)
            throw std::runtime_error("Neplatne parametry site nebo poctu kroku/snimku");
        if(p.dt<=0 || p.Du<=0 || p.Dv<=0 || p.F<0 || p.k<0 || p.iso<=0)
            throw std::runtime_error("Neplatne fyzikalni parametry");
        if(p.dt*std::max(p.Du,p.Dv)>1.0/6.0)
            throw std::runtime_error("Prilis velke dt: pro samotnou 3D difuzi plati dt*Dmax <= 1/6 (h=1)");
        const size_t count=static_cast<size_t>(p.N)*p.N*p.N;
        U.assign(count,1.f);V.assign(count,0.f);Un=U;Vn=V;
        seed();
    }
    void run() {
        std::filesystem::remove_all("frames3d");
        std::filesystem::create_directories("frames3d");
        // Metadata pro Python - automaticky sdilene parametry C++ a vizualizace.
        std::ofstream config("frames3d/config.txt");
        config<<p.N<<" "<<p.frames<<" "<<p.iso<<"\n";config.close();
        writeFrame(0);
        int saved=1;
        for(int t=1;t<=p.steps;++t) {
            step();
            // Uloz presne frames stavu, prvni je t=0, posledni t=steps.
            if(saved<p.frames && t==(saved*p.steps)/(p.frames-1)) {
                writeFrame(saved);
                std::cout<<"Snimek "<<saved+1<<"/"<<p.frames<<" (krok "<<t<<")\n";
                ++saved;
            }
        }
        // Nektere pomery steps/frames maji integer zaokrouhleni; posledni zaznam musi byt ulozen.
        if(saved<p.frames){writeFrame(saved);++saved;}
        if(saved!=p.frames) throw std::runtime_error("Nesouhlasi pocet vystupnich snimku");
        std::cout<<"Hotovo. 3D data jsou ve frames3d/.\n";
        if(p.visualize) {
            const int result=std::system("python3 visualize_3d.py");
            if(result!=0) {
                std::cerr<<"Nepodarilo se spustit vizualizaci; zkus python3 -m pip install numpy matplotlib scikit-image pillow\n";
                return;
            }
            if(p.openGif) {
                // Ve Windows/WSL otevri GIF ve vychozim prohlizeci.
                const int rc=std::system("command -v explorer.exe >/dev/null 2>&1 && explorer.exe vysledky3d/brain_coral_3d.gif >/dev/null 2>&1");
                if(rc!=0) std::cout<<"GIF: vysledky3d/brain_coral_3d.gif (otevri rucne)\n";
            }
        }
    }
};

int main(int argc,char** argv) {
    Params p;
    try {
        for(int a=1;a<argc;++a) {
            const std::string key=argv[a];
            if(key=="--no-gif") {p.visualize=false;continue;}
            if(key=="--no-open") {p.openGif=false;continue;}
            if(key=="--help") {
                std::cout<<"Pouziti: ./brain_coral_3d [--N 72] [--steps 2200] [--frames 28] "
                "[--F .035] [--k .060] [--dt .5] [--Du .18] [--Dv .09] [--iso .14] [--no-gif] [--no-open]\n";
                return 0;
            }
            if(a+1>=argc) throw std::runtime_error("Chybi hodnota argumentu " + key);
            const std::string value=argv[++a];
            if(key=="--N") p.N=std::stoi(value);
            else if(key=="--steps") p.steps=std::stoi(value);
            else if(key=="--frames") p.frames=std::stoi(value);
            else if(key=="--F") p.F=std::stod(value);
            else if(key=="--k") p.k=std::stod(value);
            else if(key=="--dt") p.dt=std::stod(value);
            else if(key=="--Du") p.Du=std::stod(value);
            else if(key=="--Dv") p.Dv=std::stod(value);
            else if(key=="--iso") p.iso=std::stod(value);
            else throw std::runtime_error("Neznamy argument: " + key);
        }
        BrainCoral coral(p);coral.run();
    } catch(const std::exception& e) {
        std::cerr<<"Chyba: "<<e.what()<<"\n";return 1;
    }
}
