#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

typedef struct GrayPair{
    uchar ref,val;
    float weight;
} graypair_t;

int hash(int _ncols, int dx, int dy, int windowsz, uint cnt)
{
    int numaddrows=cnt/windowsz, mod=cnt%windowsz;
    //if(numaddrows)printf("%d %d %d %d\n", numaddrows, mod, get_group_id(0)*(windowsz),get_group_id(1)+numaddrows);
    return _ncols*(get_group_id(0)*(windowsz)+numaddrows)+(get_group_id(1))*(windowsz)+mod;
}

int searchpair(__global graypair_t *pair, uchar ref, uchar val, int _ncols, int dx, int dy, int windowsz,uint cnt)
{
    int k;
    for(k=0;k<cnt;k++)
    {
        int addr=hash(_ncols, dx, dy, windowsz, k);
        if(pair[addr].ref==ref && pair[addr].val==val)return addr;
    }
    return -1;
}

void clearpairs(__global graypair_t *pair,int _ncols, int dx, int dy, int windowsz,uint cnt)
{
    int k;
    for(k=0;k<cnt;k++)
    {
        int addr=hash(_ncols, dx, dy, windowsz, k);
        pair[addr].ref=0;
        pair[addr].val=0;
        pair[addr].weight=0.0f;
    }
    return;
}

__kernel void glcmgen(__global uchar *img, __global float *res, __global graypair_t *pair, int dx, int dy, int windowsz, int nrows, int ncols)
{
    int i=get_global_id(0), j=get_global_id(1);
    int counts=(windowsz-dx)*(windowsz-dy);
    int hws=windowsz/2;
    //int xstart=dx>=0?hws:hws-dx, xend=dx<=0?(ncols-hws):(ncols-hws-dx);
    //int ystart=dy>=0?hws:hws-dy, yend=dy<=0?(nrows-hws):(nrows-hws-dy);
    int xstart=hws, xend=ncols-hws;
    int ystart=hws, yend=nrows-hws;
    float sum_private=0.0f;
    int stridex=(xend-xstart)/windowsz+1, stridey=(yend-ystart)/windowsz+1;
    int yrangestart=get_group_id(0)*windowsz, yrangeend=(get_group_id(0)+1)*windowsz;
    int xrangestart=get_group_id(1)*windowsz, xrangeend=(get_group_id(1)+1)*windowsz;
    //int yrangestart=6, yrangeend=7, xrangestart=6, xrangeend=7;
    __local uint cnt;
    __local float meani, meanj, vari, varj;
    meani=0.0f;meanj=0.0f;vari=0.0f;varj=0.0f;
    cnt=0;
    for(i=yrangestart;i<yrangeend;i++)
    {
        for(j=xrangestart;j<xrangeend;j++)
        {
            if(i<yend && j<xend&& i>=ystart && j>=xstart)
            {
                for(int imaddr0=(dy<0?-hws+dy:-hws);imaddr0<=(dy>0?hws-dy:hws);imaddr0++)
                {
                    for(int imaddr1=(dx<0?-hws+dx:-hws);imaddr1<=(dx>0?hws-dx:hws);imaddr1++)
                    {
                        int ref=img[(i+imaddr0)*ncols+j+imaddr1];
                        int val=img[(i+imaddr0+dy)*ncols+(imaddr1+dx+j)];
                        int tmp=searchpair(pair, min(ref, val),max(ref, val),(ncols),dx, dy, windowsz, cnt);
                        if(tmp==-1)
                        {
                            pair[hash((ncols),dx, dy, windowsz, cnt)].ref=ref;
                            pair[hash((ncols),dx, dy, windowsz, cnt)].val=val;
                            pair[hash((ncols),dx, dy, windowsz, cnt)].weight=2.0f;
                            cnt++;
                        }
                        else
                        {
                            pair[tmp].weight+=2.0f;
                        }
                    }
                }
                //dissimilarity
                for(int pairaddr0=0;pairaddr0<cnt+1;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(2*counts))*abs(pair[addr].ref-pair[addr].val);
                }
                res[0*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //contrast
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(2*counts))*(pair[addr].ref-pair[addr].val)*(pair[addr].ref-pair[addr].val);
                }
                res[1*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //homogeneity
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(2*counts))/(1+(pair[addr].ref-pair[addr].val)*(pair[addr].ref-pair[addr].val));
                }
                res[2*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //ASM
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(2*counts))*(pair[addr].weight/(2*counts));
                }
                res[3*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //energy
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(2*counts))*(pair[addr].weight/(2*counts));
                }
                res[4*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sqrt(sum_private);
                sum_private=0.0f;
                //entropy
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    float tmpres=-(pair[addr].weight/(2*counts))*log((pair[addr].weight/(2*counts)));
                    if(isnan(tmpres))continue;
                    sum_private+=tmpres;
                }
                res[5*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //Mean
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    meani+=(pair[addr].weight/(4*counts))*(pair[addr].ref+pair[addr].val);//symmetrical pairs!
                    //meanj+=(pair[addr].weight/(4*counts))*(pair[addr].ref+pair[addr].val);
                }
                meanj=meani;
                //Variance
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    vari+=(pair[addr].weight/(4*counts))*(pow((pair[addr].ref-meani),2)+pow(pair[addr].val-meanj,2));
                    //varj+=(pair[addr].weight/(4*counts))*pow((pair[addr].val-meanj),2);
                }
                varj=vari;
                //Correlation
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    //printf("%f %f %f %f\n", (pair[addr].weight/(2*counts)),(pair[addr].ref-meani),(pair[addr].val-meanj),sqrt(vari*varj));
                    sum_private+=(pair[addr].weight/(2*counts))*(((pair[addr].ref-meani)*(pair[addr].val-meanj))/(sqrt(vari*varj)));
                }
                res[6*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                meani=0;meanj=0;vari=0;varj=0;
                sum_private=0.0f;
                clearpairs(pair,(ncols),dx, dy, windowsz, ++cnt);
                cnt=0;
            }
        }
    }
    //printf("%d %d; ", get_global_id(0),get_global_id(1));
}

__kernel void glcmgen_asymmetric(__global uchar *img, __global float *res, __global graypair_t *pair, int dx, int dy, int windowsz, int nrows, int ncols)
{
    int i=get_global_id(0), j=get_global_id(1);
    int counts=(windowsz-dx)*(windowsz-dy);
    int hws=windowsz/2;
    //int xstart=dx>=0?hws:hws-dx, xend=dx<=0?(ncols-hws):(ncols-hws-dx);
    //int ystart=dy>=0?hws:hws-dy, yend=dy<=0?(nrows-hws):(nrows-hws-dy);
    int xstart=hws, xend=ncols-hws;
    int ystart=hws, yend=nrows-hws;
    float sum_private=0.0f;
    int stridex=(xend-xstart)/windowsz+1, stridey=(yend-ystart)/windowsz+1;
    int yrangestart=get_group_id(0)*windowsz, yrangeend=(get_group_id(0)+1)*windowsz;
    int xrangestart=get_group_id(1)*windowsz, xrangeend=(get_group_id(1)+1)*windowsz;
    __local uint cnt;
    cnt=0;
    __local float meani, meanj, vari, varj;
    meani=0.0f;meanj=0.0f;vari=0.0f;varj=0.0f;
    for(i=yrangestart;i<yrangeend;i++)
    {
        for(j=xrangestart;j<xrangeend;j++)
        {
            if(i<yend && j<xend&& i>=ystart && j>=xstart)
            {
                for(int imaddr0=(dy<0?-hws+dy:-hws);imaddr0<=(dy>0?hws-dy:hws);imaddr0++)
                {
                    for(int imaddr1=(dx<0?-hws+dx:-hws);imaddr1<=(dx>0?hws-dx:hws);imaddr1++)
                    {
                        int ref=img[(i+imaddr0)*ncols+j+imaddr1];
                        int val=img[(i+imaddr0+dy)*ncols+(imaddr1+dx+j)];
                        int tmp=searchpair(pair, ref,val,(ncols),dx, dy, windowsz, cnt);
                        if(tmp==-1)
                        {
                            pair[hash((ncols),dx, dy, windowsz, cnt)].ref=ref;
                            pair[hash((ncols),dx, dy, windowsz, cnt)].val=val;
                            pair[hash((ncols),dx, dy, windowsz, cnt)].weight=1.0f;
                            cnt++;
                        }
                        else
                        {
                            pair[tmp].weight+=1.0f;
                        }
                    }
                }
                //dissimilarity
                for(int pairaddr0=0;pairaddr0<cnt+1;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(counts))*abs(pair[addr].ref-pair[addr].val);
                }
                res[0*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //contrast
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(counts))*(pair[addr].ref-pair[addr].val)*(pair[addr].ref-pair[addr].val);
                }
                res[1*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //homogeneity
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    float tmpres=(pair[addr].weight/(counts))/(1+(pair[addr].ref-pair[addr].val)*(pair[addr].ref-pair[addr].val));
                    sum_private+=(pair[addr].weight/(counts))/(1+(pair[addr].ref-pair[addr].val)*(pair[addr].ref-pair[addr].val));
                }
                res[2*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                if(sum_private>1)printf("%d %d\n", get_group_id(0), get_group_id(1));
                sum_private=0.0f;
                //ASM
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(counts))*(pair[addr].weight/counts);
                }
                res[3*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //energy
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(counts))*(pair[addr].weight/counts);
                }
                res[4*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sqrt(sum_private);
                sum_private=0.0f;
                //entropy
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    float tmpres=-(pair[addr].weight/(counts))*log((pair[addr].weight/counts));
                    if(isnan(tmpres))continue;
                    sum_private+=tmpres;
                }
                res[5*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                sum_private=0.0f;
                //Mean
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    meani+=(pair[addr].weight/(counts))*(pair[addr].ref);//symmetrical pairs!
                    meanj+=(pair[addr].weight/(counts))*(pair[addr].val);
                }
                //Variance
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    vari+=(pair[addr].weight/(counts))*(pow((pair[addr].ref-meani),2));
                    varj+=(pair[addr].weight/(counts))*pow((pair[addr].val-meanj),2);
                }
                //Correlation
                for(int pairaddr0=0;pairaddr0<cnt;pairaddr0++)
                {
                    int addr=hash((ncols),dx, dy, windowsz, pairaddr0);
                    sum_private+=(pair[addr].weight/(counts))*(((pair[addr].ref-meani)*(pair[addr].val-meanj))/(sqrt(vari*varj)));
                }
                res[6*(ncols-2*hws)*(nrows-2*hws)+(i-hws)*(ncols-2*hws)+(j-hws)]=sum_private;
                meani=0;meanj=0;vari=0;varj=0;
                clearpairs(pair,(ncols),dx, dy, windowsz, cnt);
                cnt=0;
            }
        }
    }

}