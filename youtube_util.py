"""
YouTube Utility Module
Provides equipment-specific maintenance and repair videos.
"""

from typing import List, Dict, Any, Optional
import urllib.parse


EQUIPMENT_VIDEO_FALLBACKS = {
    "cnc machine": [
        {"title": "CNC Machine Maintenance Guide", "video_id": "9K3LtfI3E6o", "description": "Complete CNC maintenance procedures"},
        {"title": "CNC Programming Tutorial", "video_id": "2iLvZ9C8Vd0", "description": "CNC basics and programming"},
    ],
    "hydraulic press": [
        {"title": "Hydraulic Press Safety & Maintenance", "video_id": "rT8M7KxZ4rM", "description": "Safety and maintenance guide"},
        {"title": "Hydraulic Press Operation", "video_id": "L3Z5Yh8qY3A", "description": "Operation procedures"},
    ],
    "conveyor belt": [
        {"title": "Conveyor Belt Maintenance", "video_id": "6gR6O7eB2Fc", "description": "Preventive maintenance tips"},
        {"title": "Conveyor System Guide", "video_id": "tX2qQ3jY8Fw", "description": "System overview"},
    ],
    "industrial robot": [
        {"title": "Robot Arm Programming", "video_id": "KDs0tK7qZkA", "description": "Robot programming guide"},
        {"title": "Industrial Robot Basics", "video_id": "vL7t5KxR2Xk", "description": "Robot basics"},
    ],
    "compressor": [
        {"title": "Air Compressor Maintenance", "video_id": "qZ_U5fF4r5A", "description": "Compressor upkeep"},
        {"title": "Compressor Troubleshooting", "video_id": "N8Y7KxZ3r2M", "description": "Common issues"},
    ],
    "welding station": [
        {"title": "Welding Safety Guide", "video_id": "6sL4K8Y2x9Q", "description": "Safety procedures"},
        {"title": "Welding Basics", "video_id": "dM8Y7KxZ3r5A", "description": "Welding fundamentals"},
    ],
    "packaging machine": [
        {"title": "Packaging Equipment Maintenance", "video_id": "4hL7K8Y2x9Q", "description": "Equipment maintenance"},
    ],
    "cooling tower": [
        {"title": "Cooling Tower Maintenance", "video_id": "8pL7M9Y2x3Qk", "description": "HVAC maintenance"},
    ],
    "lathe": [
        {"title": "Lathe Machine Operation", "video_id": "5rM8K2YxZ4pA", "description": "Lathe operation"},
    ],
    "air handler": [
        {"title": "Air Handler Maintenance", "video_id": "2tP8K3YxZ4rM", "description": "HVAC maintenance"},
    ],
    "motor": [
        {"title": "Electric Motor Maintenance", "video_id": "7wR9K3YxZ2nM", "description": "Motor care"},
    ],
    "pump": [
        {"title": "Centrifugal Pump Maintenance", "video_id": "8xP9K3YxZ4qM", "description": "Pump care"},
    ],
    "fan": [
        {"title": "Industrial Fan Maintenance", "video_id": "3sM8K2YxZ4rA", "description": "Fan maintenance"},
    ],
    "gearbox": [
        {"title": "Gearbox Maintenance Guide", "video_id": "6vR9K3YxZ4nM", "description": "Gearbox repair"},
    ],
    "valve": [
        {"title": "Valve Maintenance", "video_id": "7wS8K2YxZ4pM", "description": "Valve repair"},
    ],
    "general equipment": [
        {"title": "Industrial Maintenance Basics", "video_id": "9K3LtfI3E6o", "description": "General maintenance"},
    ],
}


def extract_equipment_type(equipment_name: str) -> str:
    """Extract equipment type from full equipment name."""
    name_lower = equipment_name.lower()
    
    equipment_keywords = [
        "cnc", "machine", "hydraulic press", "conveyor", "robot", "compressor",
        "welding", "packaging", "cooling tower", "lathe", "air handler",
        "motor", "pump", "fan", "gearbox", "valve", "heater", "oven",
        "press", "mixer", "grinder", "saw", "drill", "mill"
    ]
    
    for keyword in equipment_keywords:
        if keyword in name_lower:
            if keyword in ["cnc", "machine"]:
                return "cnc machine"
            elif keyword == "hydraulic":
                return "hydraulic press"
            elif keyword == "conveyor":
                return "conveyor belt"
            elif keyword == "robot":
                return "industrial robot"
            elif keyword == "compressor":
                return "compressor"
            elif keyword == "welding":
                return "welding station"
            elif keyword == "packaging":
                return "packaging machine"
            elif keyword == "cooling":
                return "cooling tower"
            elif keyword == "lathe":
                return "lathe"
            elif keyword == "air handler":
                return "air handler"
            elif keyword == "motor":
                return "motor"
            elif keyword == "pump":
                return "pump"
            elif keyword == "fan":
                return "fan"
            elif keyword == "gearbox":
                return "gearbox"
            elif keyword == "valve":
                return "valve"
            elif keyword in ["press", "hydraulic press"]:
                return "hydraulic press"
            elif keyword in ["heater", "oven"]:
                return "general equipment"
            elif keyword in ["mixer", "grinder", "saw", "drill", "mill"]:
                return "general equipment"
    
    return "general equipment"


def get_maintenance_videos(equipment_name: str, limit: int = 4) -> List[Dict[str, str]]:
    """Get maintenance videos for specific equipment type."""
    eq_type = extract_equipment_type(equipment_name)
    
    videos = EQUIPMENT_VIDEO_FALLBACKS.get(eq_type, [])
    
    if not videos:
        videos = EQUIPMENT_VIDEO_FALLBACKS.get("general equipment", [])
    
    return videos[:limit]


def get_video_embed_html(video_id: str) -> str:
    """Get HTML iframe for embedding YouTube video."""
    return f'''
    <iframe 
        width="100%" 
        height="315" 
        src="https://www.youtube.com/embed/{video_id}" 
        frameborder="0" 
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowfullscreen>
    </iframe>
    '''


def get_video_thumbnail(video_id: str) -> str:
    """Get video thumbnail URL."""
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def get_youtube_search_url(query: str) -> str:
    """Get YouTube search URL."""
    encoded = urllib.parse.quote(query)
    return f"https://www.youtube.com/results?search_query={encoded}"


