import google.generativeai as genai
import json
from typing import List, Dict, Optional
import re
import requests
import json
class VideoPromptRefiner:
    
    def __init__(self, api_key: str, model_name: str = "tngtech/deepseek-r1t2-chimera"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        
    def refine_prompts_for_sections(self, 
                                  original_prompt: str, 
                                  num_sections: int,
                                  frames_per_section: int = 70,
                                  overlap_frames: int = 2,
                                  context_info: Optional[Dict] = None) -> Dict[str, any]:
        
        system_instruction = self._build_system_instruction(
            num_sections, frames_per_section, overlap_frames
        )
        
        user_prompt = self._build_user_prompt(original_prompt, num_sections, context_info)
        
        try:
            # DeepSeek API call
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
            print(response_text)
            
        except Exception as e:
            print(f"Error refining prompts: {e}")
            return self._fallback_prompt_division(original_prompt, num_sections)
    
    def _build_system_instruction(self, num_sections: int, frames_per_section: int, overlap_frames: int) -> str:
        """Build the system instruction for the LLM."""
        return f"""You are a video generation prompt specialist. Your task is to refine and divide a video generation prompt into {num_sections} sequential sections that maintain continuity and coherence.

Each section will generate {frames_per_section} frames with {overlap_frames} overlap frames for smooth transitions.

Your response must be a valid JSON object with this exact structure:
{{
    "global_style": "consistent style descriptors that apply to all sections",
    "narrative_arc": "brief description of the overall story progression",
    "sections": [
        {{
            "section_id": 0,
            "prompt": "detailed prompt for this section",
            "key_elements": ["element1", "element2"],
            "transition_hint": "how this connects to next section",
            "camera_movement": "camera behavior in this section",
            "time_of_day": "lighting/time context",
            "emotion": "emotional tone"
        }},
        // ... more sections
    ],
    "consistency_notes": "important elements to maintain across all sections"
}}

Guidelines:
1. Each section prompt should be self-contained but reference previous context
2. Include specific details about motion, lighting, and atmosphere
3. Ensure smooth narrative progression between sections
4. Maintain consistent character/object descriptions
5. Add temporal markers (beginning, middle, end) when relevant
6. Include camera movement descriptions for cinematic flow
7. Preserve the original prompt's intent while adding helpful details

Focus on creating prompts that will generate a cohesive video when concatenated."""

    def _fallback_prompt_division(self, original_prompt: str, num_sections: int) -> Dict:
        """Simple fallback prompt division if LLM fails."""
        base_prompt = original_prompt.strip()
        
        sections = []
        for i in range(num_sections):
            if i == 0:
                section_prompt = f"{base_prompt}. Beginning of sequence."
            elif i == num_sections - 1:
                section_prompt = f"{base_prompt}. Final part of sequence, bringing closure."
            else:
                section_prompt = f"{base_prompt}. Continuing the sequence (part {i+1}/{num_sections})."
            
            sections.append({
                'section_id': i,
                'prompt': section_prompt,
                'key_elements': [base_prompt],
                'transition_hint': 'continue naturally'
            })
        
        return {
            'global_style': 'consistent throughout',
            'sections': sections,
            'narrative_arc': 'continuous sequence',
            'consistency_notes': 'Fallback division - maintain all original elements'
        }