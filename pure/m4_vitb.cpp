// Staged parity: pure-C++ SAM ViT-B encoder vs PyTorch. (Slow: global blocks attend over 4096 tokens.)
#include "net_vitb.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string& p, size_t n){ std::vector<float> v(n); std::ifstream f(p,std::ios::binary); if(!f){printf("missing %s\n",p.c_str());std::exit(1);} f.read((char*)v.data(),n*4); return v; }
static float wr(const std::vector<float>& a, const std::vector<float>& b){ float w=0; size_t n=std::min(a.size(),b.size()); for(size_t i=0;i<n;++i) w=std::max(w,std::fabs(a[i]-b[i])); return w; }
int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  VitbW w=load_vitb(RF);
  Tensor img=from_data({1,3,1024,1024}, rd(RF+"vitb_in.bin",3*1024*1024));
  printf("running ViT-B (slow)...\n");
  Tensor emb=vitb_forward(img,w);
  printf("pe+pos  worst = %.3e\n", wr(g_vb_pe->data, rd(RF+"vitb_pe_pos.bin",4096*768)));
  printf("block0  worst = %.3e\n", wr(g_vb_b0->data, rd(RF+"vitb_block0.bin",4096*768)));
  printf("block2  worst = %.3e  (first global)\n", wr(g_vb_b2->data, rd(RF+"vitb_block2.bin",4096*768)));
  printf("block11 worst = %.3e\n", wr(g_vb_b11->data, rd(RF+"vitb_block11.bin",4096*768)));
  auto fr=rd(RF+"vitb_final.bin",256*64*64); float fw=wr(emb->data,fr);
  double mref=0; for(float v:fr)mref+=std::fabs(v); mref/=fr.size();
  printf("final   worst = %.3e  (mean|ref| %.3f)  %s\n", fw, mref, fw<2e-3?"MATCH":"MISMATCH");
  return 0;
}
