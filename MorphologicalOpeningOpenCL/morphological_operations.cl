__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void morphological_operations(
        __read_only image2d_t image,
        __global float * processedImage, 
		__private int dilationOrErosion
    ) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
    
	// centre pixel
    float cPoint = read_imagef(image, sampler, pos ).x;
	
	// centreX + 1
	float cPointXP1 = read_imagef(image, sampler, pos + (int2)(1,0)).x;
	
	// centreY + 1
	float cPointYP1 = read_imagef(image, sampler, pos + (int2)(0,1)).x;
	
	//centreX - 1
	float cPointXM1 = read_imagef(image, sampler, pos - (int2)(1,0)).x;
	
	// centreY - 1
	float cPointYM1 = read_imagef(image, sampler, pos - (int2)(0,1)).x;
	
	if(dilationOrErosion == 0)
		processedImage[pos.x+pos.y*get_global_size(0)] = max( cPoint, max( max(cPointXP1, cPointXM1), max(cPointYP1, cPointYM1) ) );
	else
		processedImage[pos.x+pos.y*get_global_size(0)] = min( cPoint, min( min(cPointXP1, cPointXM1), min(cPointYP1, cPointYM1) ) );
}