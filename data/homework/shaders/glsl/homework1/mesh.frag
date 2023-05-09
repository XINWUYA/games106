#version 450

layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;
layout (set = 1, binding = 1) uniform sampler2D samplerNormalMap;
layout (set = 1, binding = 2) uniform sampler2D samplerMetallicRoughnessMap;
layout (set = 1, binding = 3) uniform sampler2D samplerOcclusionMap;
layout (set = 1, binding = 4) uniform sampler2D samplerEmissiveMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout (location = 5) in vec4 inTangent;

layout (location = 0) out vec4 outFragColor;

layout (constant_id = 0) const bool ALPHA_MASK = false;
layout (constant_id = 1) const float ALPHA_MASK_CUTOFF = 0.0f;

//layout(binding = 0) uniform FragUniforms {
//	vec4 ambientFactor;
//} fragUniforms;

const float PI = 3.14159265359;

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom);
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, float metallic, vec3 albedo)
{
	vec3 F0 = mix(vec3(0.04), albedo, metallic);
	vec3 F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
	return F;
}

// Specular BRDF composition --------------------------------------------

vec3 BRDF(vec3 L, vec3 V, vec3 N, float metallic, float roughness, vec3 albedo)
{
	// Precalculate vectors and dot products	
	vec3 H = normalize (V + L);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);
	float dotLH = clamp(dot(L, H), 0.0, 1.0);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);

	// Light color fixed
	vec3 lightColor = vec3(1.0);

	vec3 color = vec3(0.0);

	if (dotNL > 0.0)
	{
		float rroughness = max(0.05, roughness);
		// D = Normal distribution (Distribution of the microfacets)
		float D = D_GGX(dotNH, roughness);
		// G = Geometric shadowing term (Microfacets shadowing)
		float G = G_SchlicksmithGGX(dotNL, dotNV, rroughness);
		// F = Fresnel factor (Reflectance depending on angle of incidence)
		vec3 F = F_Schlick(dotNV, metallic, albedo);
		
		vec3 kD = vec3(1.0) - F;
		vec3 diff = kD * (1.0 - metallic) * albedo / PI;
		vec3 spec = D * F * G / (4.0 * dotNL * dotNV);

		color += (diff + spec) * dotNL * lightColor;
	}

	return color;
}

vec3 SrgbToLinear(vec3 color)
{
	return pow(color, vec3(2.2));
}

// tonemap
vec3 Tonemap_ACES(const vec3 c)
{
	// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
	// const float a = 2.51;
	// const float b = 0.03;
	// const float c = 2.43;
	// const float d = 0.59;
	// const float e = 0.14;
	// return saturate((x*(a*x+b))/(x*(c*x+d)+e));

	//ACES RRT/ODT curve fit courtesy of Stephen Hill
	vec3 a = c * (c + 0.0245786) - 0.000090537;
	vec3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
	return a / b;
}

void main() 
{
	vec4 color = texture(samplerColorMap, inUV);
	vec4 normal = texture(samplerNormalMap, inUV);
	vec4 metallicRoughness = texture(samplerMetallicRoughnessMap, inUV);
	vec4 emissive = texture(samplerEmissiveMap, inUV);
	float occlusion = texture(samplerOcclusionMap, inUV).r;

	if (ALPHA_MASK) {
		if (color.a < ALPHA_MASK_CUTOFF) {
			discard;
		}
	}
	
	vec3 T = normalize(inTangent.xyz);
	vec3 B = cross(inNormal, inTangent.xyz) * inTangent.w;
	vec3 N = normalize(inNormal);
	mat3 TBN = mat3(T, B, N);
	N = TBN * normalize(texture(samplerNormalMap, inUV).xyz * 2.0 - vec3(1.0));
	
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = reflect(L, N);
	
	vec3 albedo = SrgbToLinear(color.rgb) * inColor;
	// Direct 
	vec3 lightingResult = max(vec3(0.0f), BRDF(L, V, N, metallicRoughness.r, metallicRoughness.g, albedo));
	// Ambient
	vec3 ambient = albedo * occlusion * 0.5f;// * fragUniforms.ambientFactor.rgb;
	lightingResult += max(vec3(0.0f), ambient);
	// Emissive
	lightingResult += max(vec3(0.0f), emissive.rgb);
	
	outFragColor = vec4(Tonemap_ACES(lightingResult), color.a);
	//outFragColor = vec4(normal.rgb, 1.0);
}