// Staged parity: pure-C++ TinyViT encoder vs PyTorch mobile_sam image_encoder.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m2_tinyvit.cpp
#include "net_tinyvit.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
static std::vector<float> rd(const std::string& p, size_t n){ std::vector<float> v(n); std::ifstream f(p,std::ios::binary); if(!f){printf("missing %s\n",p.c_str());std::exit(1);} f.read((char*)v.data(),n*4); return v; }
static float wr(const std::vector<float>& a, const std::vector<float>& b){ float w=0; size_t n=std::min(a.size(),b.size()); for(size_t i=0;i<n;++i) w=std::max(w,std::fabs(a[i]-b[i])); return w; }
int main(int argc,char**argv){
  std::string RF=argc>1?argv[1]:"pure/ref/"; if(RF.back()!='/')RF+='/';
  TvitW w=load_tinyvit(RF);
  Tensor img=from_data({1,3,1024,1024}, rd(RF+"tvit_in.bin",3*1024*1024));
  printf("running TinyViT...\n"); fflush(stdout);
  Tensor emb=tinyvit_forward(img,w);
  printf("patch_embed worst = %.3e\n", wr(g_tv_pe->data, rd(RF+"tvit_patch_embed.bin",64*256*256)));
  printf("layer0      worst = %.3e\n", wr(g_tv_l0->data, rd(RF+"tvit_layer0.bin",16384*128)));
  printf("layer1      worst = %.3e\n", wr(g_tv_l1->data, rd(RF+"tvit_layer1.bin",4096*160)));
  printf("layer2      worst = %.3e\n", wr(g_tv_l2->data, rd(RF+"tvit_layer2.bin",4096*320)));
  printf("layer3      worst = %.3e\n", wr(g_tv_l3->data, rd(RF+"tvit_layer3.bin",4096*320)));
  auto fr=rd(RF+"tvit_final.bin",256*64*64); float fw=wr(emb->data,fr);
  double mref=0; for(float v:fr)mref+=std::fabs(v); mref/=fr.size();
  printf("final embed  worst = %.3e  (mean|ref| %.3f)  %s\n", fw, mref, fw<1e-3?"MATCH":"MISMATCH");
  return 0;
}
