#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

// Growing circular domain with Gray-Scott reaction-diffusion dynamics.
// Finite differences in 2D, forward Euler in time, no-flux boundary.
struct Parameters {
    int N=256, steps=9500, frames=48, seed=42;
    double Du=0.16, Dv=0.08, F=0.037, k=0.060, dt=1.0;
    double startRadius=24.0, endRadius=114.0;
    std::string output="results";
};

class Coral {
    Parameters p;
    std::vector<double> u,v,un,vn;
    std::vector<unsigned char> active;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform{0.0,1.0};
    double radius=0;
    int index(int x,int y)const{return y*p.N+x;}
    bool inside(int x,int y)const{return x>=0&&x<p.N&&y>=0&&y<p.N&&active[index(x,y)];}
    double neighbor(const std::vector<double>& field,int x,int y,int fallback)const{
        return inside(x,y)?field[index(x,y)]:field[fallback];
    }
public:
    explicit Coral(Parameters pp):p(std::move(pp)),u(p.N*p.N,1),v(p.N*p.N,0),un(u),vn(v),active(p.N*p.N,0),rng(p.seed) {
        if(p.N<64||p.steps<1||p.frames<2||p.dt<=0||p.startRadius<3||p.endRadius>=p.N/2.0-2)
            throw std::invalid_argument("Invalid simulation parameters");
        if(p.dt*4*std::max(p.Du,p.Dv)>1.0)
            throw std::invalid_argument("Time step may violate explicit diffusion stability bound");
        updateDomain(p.startRadius);
        // Locally inject v; not a drawn line pattern.
        for(int n=0;n<28;++n){
            double angle=2*3.14159265358979323846*uniform(rng);
            double rr=(0.12+0.75*std::sqrt(uniform(rng)))*p.startRadius;
            int cx=static_cast<int>(std::lround(p.N/2.0+rr*std::cos(angle)));
            int cy=static_cast<int>(std::lround(p.N/2.0+rr*std::sin(angle)));
            for(int dy=-3;dy<=3;++dy)for(int dx=-3;dx<=3;++dx)
                if(dx*dx+dy*dy<=9&&inside(cx+dx,cy+dy)) {
                    int q=index(cx+dx,cy+dy);u[q]=0.45;v[q]=0.25+0.1*uniform(rng);
                }
        }
    }
    void updateDomain(double nextRadius){
        radius=nextRadius;
        std::vector<unsigned char> old=active;
        for(int y=0;y<p.N;++y)for(int x=0;x<p.N;++x){
            double dx=(x+.5)-p.N/2.0,dy=(y+.5)-p.N/2.0;
            int i=index(x,y);if(dx*dx+dy*dy>radius*radius)continue;
            active[i]=1;
            if(old[i])continue;
            // New tissue inherits concentration from established neighbors.
            double su=0,sv=0;int count=0;
            for(int oy=-1;oy<=1;++oy)for(int ox=-1;ox<=1;++ox){
                int xx=x+ox,yy=y+oy;
                if(xx>=0&&xx<p.N&&yy>=0&&yy<p.N&&old[index(xx,yy)]){
                    int j=index(xx,yy);su+=u[j];sv+=v[j];++count;
                }
            }
            if(count){u[i]=su/count;v[i]=sv/count;}
            else {u[i]=1;v[i]=0;}
        }
    }
    void step(){
        for(int y=0;y<p.N;++y)for(int x=0;x<p.N;++x){
            const int i=index(x,y);
            if(!active[i])continue;
            double a=u[i],b=v[i],r=a*b*b;
            double Lu=(neighbor(u,x-1,y,i)+neighbor(u,x+1,y,i)+neighbor(u,x,y-1,i)+neighbor(u,x,y+1,i)-4*a);
            double Lv=(neighbor(v,x-1,y,i)+neighbor(v,x+1,y,i)+neighbor(v,x,y-1,i)+neighbor(v,x,y+1,i)-4*b);
            un[i]=std::clamp(a+p.dt*(p.Du*Lu-r+p.F*(1-a)),0.0,1.0);
            vn[i]=std::clamp(b+p.dt*(p.Dv*Lv+r-(p.F+p.k)*b),0.0,1.0);
        }
        u.swap(un);v.swap(vn);
    }
    void save(int t)const{
        std::filesystem::create_directories(p.output);
        std::ostringstream name; name<<p.output<<"/coral_"<<std::setw(5)<<std::setfill('0')<<t<<".ppm";
        std::ofstream out(name.str(),std::ios::binary);
        if(!out)throw std::runtime_error("Cannot write output file: "+name.str());
        out<<"P6\n"<<p.N<<' '<<p.N<<"\n255\n";
        for(int i=0;i<p.N*p.N;++i){
            unsigned char rgb[3]={250,249,246};
            if(active[i]){
                // Dark gray ridges from concentration of the v species.
                double z=std::clamp((v[i]-0.045)/0.28,0.0,1.0);
                z=z*z*(3-2*z);
                for(int k=0;k<3;++k)rgb[k]=static_cast<unsigned char>((1-z)*rgb[k]+z*(k==0?38:(k==1?53:61)));
            }
            out.write(reinterpret_cast<char*>(rgb),3);
        }
        if(!out)throw std::runtime_error("Output write error");
        std::cout<<"step="<<t<<" radius="<<std::fixed<<std::setprecision(1)<<radius<<" saved "<<name.str()<<'\n';
    }
    void run(){
        save(0);
        for(int t=1;t<=p.steps;++t){
            // Boundary expands at a constant rate; updated less frequently to avoid jagged initialization.
            if(t%5==0)updateDomain(p.startRadius+(p.endRadius-p.startRadius)*t/p.steps);
            step();
            if((static_cast<long long>(t)*p.frames/p.steps) != (static_cast<long long>(t-1)*p.frames/p.steps) || t==p.steps) save(t);
        }
    }
};

int main(int argc,char** argv){
    try {
        Parameters p;
        for(int i=1;i<argc;++i){
            std::string key=argv[i];
            if(key=="--help" || key=="-h"){
                std::cout << "Usage: ./coral_labyrinth [--F 0.037] [--k 0.060] [--seed 42] "
                             "[--steps 9500] [--frames 48] [--output results]\n";
                return 0;
            }
            if(i+1>=argc) throw std::invalid_argument("Missing value for "+key);
            std::string value=argv[++i];
            if(key=="--F" || key=="--f") p.F=std::stod(value);
            else if(key=="--k") p.k=std::stod(value);
            else if(key=="--seed") p.seed=std::stoi(value);
            else if(key=="--steps") p.steps=std::stoi(value);
            else if(key=="--frames") p.frames=std::stoi(value);
            else if(key=="--output") p.output=value;
            else throw std::invalid_argument("Unknown option: "+key);
        }
        if(!std::isfinite(p.F)||!std::isfinite(p.k)||p.F<=0||p.F>=1||p.k<=0||p.k>=1)
            throw std::invalid_argument("F and k must be finite and in (0,1)");
        if(p.frames>p.steps) throw std::invalid_argument("frames must not exceed steps");
        std::cout << "F="<<p.F<<" k="<<p.k<<" steps="<<p.steps
                  <<" frames="<<p.frames<<" seed="<<p.seed<<'\n';
        Coral simulation(p);
        simulation.run();
    } catch(const std::exception& e){
        std::cerr << "Error: "<<e.what()<<'\n';
        return 1;
    }
}
