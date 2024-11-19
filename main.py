from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
import json
from itertools import product

class AdvancedCropVariety(BaseModel):
    """
    Enhanced Pydantic model with comprehensive field mapping
    """
    sl: Optional[str] = Field(None, alias="SL")
    name: Optional[str] = Field(None, alias="Name of Variety")
    parentage: Optional[str] = Field(None, alias="Parentage")
    crop_type: Optional[str] = Field(None, alias="")
    eco_system: Optional[str] = Field(None, alias="Eco-System")
    salient_features: Optional[str] = Field(None, alias="Salient Features")
    states: Optional[str] = Field(None, alias="States")

class AdvancedCropSearch:
    def __init__(self, crop_data: List[Dict]):
        """
        Initialize search service with comprehensive crop data
        
        Args:
            crop_data (List[Dict]): Full list of crop variety data
        """
        self.crop_data = crop_data

    def advanced_search(
        self,
        crop_type: Optional[str] = None,
        state: Optional[str] = None,
        variety_name: Optional[str] = None,
        parentage: Optional[str] = None,
        eco_system: Optional[str] = None,
        salient_features: Optional[str] = None,
        min_results: Optional[int] = None,
        max_results: Optional[int] = None,
        exact_match: bool = False
    ) -> List[Dict]:
        """
        Comprehensive advanced search with multiple filtering options
        
        Args:
            crop_type (Optional[str]): Crop type filter
            state (Optional[str]): State filter
            variety_name (Optional[str]): Variety name filter
            parentage (Optional[str]): Parentage filter
            eco_system (Optional[str]): Ecosystem filter
            salient_features (Optional[str]): Salient features filter
            min_results (Optional[int]): Minimum number of results
            max_results (Optional[int]): Maximum number of results
            exact_match (bool): Whether to use exact matching
        
        Returns:
            List of matching crop varieties
        """
        def check_match(item: Dict, field: str, value: str, exact: bool = False) -> bool:
            """
            Flexible matching for different search criteria
            
            Args:
                item (Dict): Individual crop variety item
                field (str): Field to search in
                value (str): Search value
                exact (bool): Whether to use exact matching
            
            Returns:
                bool: Whether the item matches the criteria
            """
            if not value:
                return True
            
            item_value = str(item.get(field, '')).lower().strip()
            value = value.lower().strip()
            
            if exact_match:
                return item_value == value
            
            # Support multiple state/crop matching
            if field in ['States', '']:
                return any(
                    value in state.lower().strip() 
                    for state in item_value.split(',')
                )
            
            return value in item_value

        # Apply all filters
        filtered_results = [
            item for item in self.crop_data
            if (check_match(item, '', crop_type, exact_match) if crop_type else True) and
               (check_match(item, 'States', state, exact_match) if state else True) and
               (check_match(item, 'Name of Variety', variety_name, exact_match) if variety_name else True) and
               (check_match(item, 'Parentage', parentage, exact_match) if parentage else True) and
               (check_match(item, 'Eco-System', eco_system, exact_match) if eco_system else True) and
               (check_match(item, 'Salient Features', salient_features, exact_match) if salient_features else True)
        ]
        
        # Apply result limits
        if min_results:
            filtered_results = filtered_results[:min_results]
        if max_results:
            filtered_results = filtered_results[:max_results]
        
        return filtered_results

    def complex_combination_search(
        self, 
        crop_types: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        search_strategy: str = 'AND'
    ) -> List[Dict]:
        """
        Advanced combination search with multiple crop types and states
        
        Args:
            crop_types (Optional[List[str]]): List of crop types to search
            states (Optional[List[str]]): List of states to search
            search_strategy (str): Search strategy - 'AND' or 'OR'
        
        Returns:
            List of matching crop varieties
        """
        if not crop_types and not states:
            return self.crop_data
        
        def sanitize_input(input_list: Optional[List[str]]) -> List[str]:
            """
            Sanitize and normalize input list
            
            Args:
                input_list (Optional[List[str]]): Input list to sanitize
            
            Returns:
                List of sanitized strings
            """
            return [
                item.lower().strip() 
                for item in (input_list or []) 
                if item and item.strip()
            ]
        
        crop_types = sanitize_input(crop_types)
        states = sanitize_input(states)
        
        def item_matches(item: Dict) -> bool:
            """
            Check if an item matches the combination search criteria
            
            Args:
                item (Dict): Individual crop variety item
            
            Returns:
                bool: Whether the item matches the search criteria
            """
            item_crop = str(item.get('', '')).lower().strip()
            item_states = [
                state.lower().strip() 
                for state in item.get('States', '').split(',')
            ]
            
            crop_match = not crop_types or any(
                crop in item_crop for crop in crop_types
            )
            
            state_match = not states or any(
                any(state in item_state for state in states) 
                for item_state in item_states
            )
            
            return (
                (crop_match and state_match) if search_strategy == 'AND'
                else (crop_match or state_match)
            )
        
        return [item for item in self.crop_data if item_matches(item)]

    def get_comprehensive_stats(self) -> Dict[str, Union[int, List[str]]]:
        """
        Generate comprehensive statistics about the crop varieties
        
        Returns:
            Dict with various statistical insights
        """
        unique_crops = set(
            item.get('', '').strip() 
            for item in self.crop_data 
            if item.get('', '').strip()
        )
        
        unique_states = set()
        for item in self.crop_data:
            states = item.get('States', '').split(',')
            unique_states.update(
                state.strip() 
                for state in states 
                if state.strip()
            )
        
        return {
            'total_varieties': len(self.crop_data),
            'unique_crops': list(unique_crops),
            'unique_states': list(unique_states),
            'crop_distribution': {
                crop: sum(1 for item in self.crop_data if item.get('', '').strip() == crop)
                for crop in unique_crops
            }
        }

# FastAPI Application Setup
app = FastAPI(
    title="Advanced Crop Variety Search API",
    description="Comprehensive API for advanced crop variety searching and analysis"
)

# Load your JSON data (replace with actual path)
def load_crop_data(file_path: str) -> List[Dict]:
    """
    Load crop variety data from JSON file
    
    Args:
        file_path (str): Path to JSON file
    
    Returns:
        List of crop variety dictionaries
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

# Initialize crop data and search service
CROP_DATA = load_crop_data('seed-varitiess.json')
crop_search_service = AdvancedCropSearch(CROP_DATA)

@app.get("/advanced-search", response_model=List[AdvancedCropVariety])
async def advanced_search(
    crop_type: Optional[str] = Query(None, description="Crop type"),
    state: Optional[str] = Query(None, description="State"),
    variety_name: Optional[str] = Query(None, description="Variety Name"),
    parentage: Optional[str] = Query(None, description="Parentage"),
    eco_system: Optional[str] = Query(None, description="Ecosystem"),
    salient_features: Optional[str] = Query(None, description="Salient Features"),
    min_results: Optional[int] = Query(None, description="Minimum number of results"),
    max_results: Optional[int] = Query(None, description="Maximum number of results"),
    exact_match: bool = Query(False, description="Use exact matching")
):
    """
    Advanced search endpoint with multiple filtering options
    
    Supports complex querying with optional parameters
    """
    results = crop_search_service.advanced_search(
        crop_type, state, variety_name, parentage, 
        eco_system, salient_features, 
        min_results, max_results, exact_match
    )
    
    if not results:
        raise HTTPException(status_code=404, detail="No matching varieties found")
    
    return results

@app.get("/combination-search", response_model=List[AdvancedCropVariety])
async def combination_search(
    crop_types: Optional[List[str]] = Query(None, description="List of crop types"),
    states: Optional[List[str]] = Query(None, description="List of states"),
    search_strategy: str = Query('AND', description="Search strategy: 'AND' or 'OR'")
):
    """
    Advanced combination search with multiple crop types and states
    
    Supports complex multi-crop and multi-state searching
    """
    results = crop_search_service.complex_combination_search(
        crop_types, states, search_strategy
    )
    
    if not results:
        raise HTTPException(status_code=404, detail="No matching varieties found")
    
    return results

@app.get("/crop-stats")
async def get_crop_statistics():
    """
    Retrieve comprehensive crop variety statistics
    """
    return crop_search_service.get_comprehensive_stats()

# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)