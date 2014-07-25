typedef struct 
 {
   unsigned char Blue; // Blue channel  
   unsigned char Green; // Green channel
   unsigned char Red; // Red channel
 }RGBValue;

void RGB2HSV(int r, int g, int b, float *fh, float *fs, float *fv, __local int *div_table)
{
      // mostly copied from opencv-svn/modules/imgproc/src/color.cpp
      // revision is 4351
      int hsv_shift = 12;
      int hr = 180, hscale = 15;
      int h, s, v = b;
      int vmin = b, diff;
      int vr, vg;
                    
      v = max(v, g);
      v = max (v, r);
      vmin = min(vmin, g);
      vmin = min(vmin, r);
                
      diff = v - vmin;
      vr = v == r ? -1 : 0;
      vg = v == g ? -1 : 0;
                    
      s = diff * div_table[v] >> hsv_shift;
      h = (vr & (g - b)) +
          (~vr & ((vg & (b - r + 2 * diff))
          + ((~vg) & (r - g + 4 * diff))));
      h = (h * div_table[diff] * hscale +
          (1 << (hsv_shift + 6))) >> (7 + hsv_shift);
                
      h += h < 0 ? hr : 0;
      *fh = (h) / 180.0f;
      *fs = (s) / 255.0f;
      *fv = (v) / 255.0f;
}
 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void HSVcomputeCoherence(__global RGBValue *source, __global RGBValue *target, __global float *weight, int size, __global int * div_data, float h_weight_, float s_weight_, float v_weight_, float weight_)
{

      int thread = get_global_id(0);
      __local int div_table[256];
      
      if(thread < 256)
        div_table[thread] = div_data[thread]; 
	
      if(thread < size)
      {
      // convert color space from RGB to HSV
      RGBValue source_rgb, target_rgb;
      source_rgb = source[thread];
      target_rgb = target[thread];

      float source_h, source_s, source_v, target_h, target_s, target_v;
      RGB2HSV (source_rgb.Red, source_rgb.Blue, source_rgb.Green, &source_h, &source_s, &source_v, div_table);
      RGB2HSV (target_rgb.Red, target_rgb.Blue, target_rgb.Green, &target_h, &target_s, &target_v, div_table);
      // hue value is in 0 ~ 2pi, but circulated.
      float _h_diff = fabs(source_h - target_h);
      // Also need to compute distance other way around circle - but need to check which is closer to 0
      float _h_diff2;
      if (source_h < target_h)
        _h_diff2 = fabs(1.0f + source_h - target_h); //Add 2pi to source, subtract target
      else 
        _h_diff2 = fabs(1.0f + target_h - source_h); //Add 2pi to target, subtract source
      
      float h_diff;
      //Now we need to choose the smaller distance
      if (_h_diff < _h_diff2)
        h_diff = (float)(h_weight_) * _h_diff * _h_diff;
      else
        h_diff = (float)(h_weight_) * _h_diff2 * _h_diff2;

      float s_diff = (s_weight_) * (source_s - target_s) * (source_s - target_s);
      float v_diff = (v_weight_) * (source_v - target_v) * (source_v - target_v);
      float diff2 = h_diff + s_diff + v_diff;
      
      weight[thread] = (1.0 / (1.0 + weight_ * diff2));
      }
}
  


