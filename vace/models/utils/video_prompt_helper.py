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
            
            refined_data = self._parse_llm_response(response_text, num_sections)
            
            refined_data['original_prompt'] = original_prompt
            refined_data['num_sections'] = num_sections
            refined_data['frames_per_section'] = frames_per_section
            
            return refined_data
            
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

    def _build_user_prompt(self, original_prompt: str, num_sections: int, context_info: Optional[Dict]) -> str:
        """Build the user prompt for the LLM."""
        prompt = f"Original video prompt: \"{original_prompt}\"\n\n"
        prompt += f"Please divide this into {num_sections} sections for sequential video generation.\n"
        
        if context_info:
            prompt += "\nAdditional context:\n"
            for key, value in context_info.items():
                prompt += f"- {key}: {value}\n"
        
        prompt += "\nRemember to maintain continuity between sections and enhance the original prompt with specific details that will help generate a coherent video."
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, num_sections: int) -> Dict:
        """Parse the LLM response into structured data."""
        try:
           
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
              
                if self._validate_response_structure(data, num_sections):
                    return data
                else:
                    print("Invalid response structure, using parsed data with corrections")
                    return self._correct_response_structure(data, num_sections)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            
            return self._extract_prompts_manually(response_text, num_sections)
    
    def _validate_response_structure(self, data: Dict, num_sections: int) -> bool:
        """Validate the parsed response has the expected structure."""
        required_keys = ['global_style', 'sections']
        if not all(key in data for key in required_keys):
            return False
        
        if len(data.get('sections', [])) != num_sections:
            return False
        
        for section in data['sections']:
            if 'prompt' not in section:
                return False
        
        return True
    
    def _correct_response_structure(self, data: Dict, num_sections: int) -> Dict:
        
        if 'global_style' not in data:
            data['global_style'] = "cinematic, high quality"
        
        if 'sections' not in data:
            data['sections'] = []
        
       
        while len(data['sections']) < num_sections:
            data['sections'].append({
                'section_id': len(data['sections']),
                'prompt': f"Continue the video sequence (section {len(data['sections']) + 1})",
                'key_elements': [],
                'transition_hint': 'smooth continuation'
            })
      
        for i, section in enumerate(data['sections'][:num_sections]):
            if 'section_id' not in section:
                section['section_id'] = i
            if 'prompt' not in section:
                section['prompt'] = f"Section {i + 1} of the video"
            if 'key_elements' not in section:
                section['key_elements'] = []
        
        return data
    
    def _extract_prompts_manually(self, response_text: str, num_sections: int) -> Dict:
        """Manually extract prompts from response text if JSON parsing fails."""
        lines = response_text.split('\n')
        sections = []
        current_section = None
        
        for line in lines:
            if 'section' in line.lower() and ('prompt:' in line.lower() or ':' in line):
                if current_section and 'prompt' in current_section:
                    sections.append(current_section)
                current_section = {
                    'section_id': len(sections),
                    'prompt': line.split(':', 1)[-1].strip() if ':' in line else line,
                    'key_elements': []
                }
            elif current_section and line.strip() and not line.strip().startswith('{'):
                current_section['prompt'] += ' ' + line.strip()
        
        if current_section:
            sections.append(current_section)
        
        # Ensure we have the right number of sections
        while len(sections) < num_sections:
            sections.append({
                'section_id': len(sections),
                'prompt': f"Continue section {len(sections) + 1}",
                'key_elements': []
            })
        
        return {
            'global_style': 'extracted from response',
            'sections': sections[:num_sections],
            'consistency_notes': 'Manually extracted prompts'
        }
    
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
    
    def format_section_prompt(self, section_data: Dict, global_style: str) -> str:
        """
        Format a section's data into a complete prompt for video generation.
        
        Args:
            section_data: Dictionary containing section information
            global_style: Global style to prepend
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [global_style, section_data['prompt']]
        
        # Add optional elements if they exist
        if 'camera_movement' in section_data:
            prompt_parts.append(f"Camera: {section_data['camera_movement']}")
        
        if 'time_of_day' in section_data:
            prompt_parts.append(f"Time: {section_data['time_of_day']}")
        
        if 'emotion' in section_data:
            prompt_parts.append(f"Mood: {section_data['emotion']}")
        
        return '. '.join(filter(None, prompt_parts))


def use_prompt_refiner(api_key: str, video_prompt: str, num_sections: int):
    """Example of how to use the VideoPromptRefiner."""
    
    
    refiner = VideoPromptRefiner(api_key)
    
   
    context_info = {
        "video_length": f"{num_sections * 70} frames total",
        "style_preference": "cinematic and smooth",
        "target_mood": "epic and inspiring"
    }
    
   
    refined_data = refiner.refine_prompts_for_sections(
        original_prompt=video_prompt,
        num_sections=num_sections,
        frames_per_section=70,
        overlap_frames=2,
        context_info=context_info
    )
    
    
    print(f"Original prompt: {refined_data['original_prompt']}")
    print(f"\nGlobal style: {refined_data.get('global_style', 'N/A')}")
    print(f"Narrative arc: {refined_data.get('narrative_arc', 'N/A')}")
    print(f"\nSection prompts:")
    
    for section in refined_data['sections']:
        print(f"\n--- Section {section['section_id'] + 1} ---")
        formatted_prompt = refiner.format_section_prompt(
            section, 
            refined_data.get('global_style', '')
        )
        print(f"Prompt: {formatted_prompt}")
        if 'transition_hint' in section:
            print(f"Transition: {section['transition_hint']}")
    
    return refined_data
