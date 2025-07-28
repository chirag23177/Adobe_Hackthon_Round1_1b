# Document Intelligence - JSON Input Support

## Overview
The Document Intelligence system has been updated to accept input in JSON format and process PDFs from a dedicated PDFs folder.

## Changes Made

### 1. JSON Input Format
- The system now reads persona, job to be done, and PDF file list from a JSON input file
- Default input file: `./input/challenge1.json`
- Can specify custom input file via command line argument

### 2. PDF Processing
- PDFs are now read from the `./PDFs` folder instead of the input folder
- Only processes PDFs specified in the JSON input file
- Provides clear feedback on which files are found/missing

### 3. Command Line Support
- Can specify different input JSON files as command line arguments
- Usage: `python main.py [path_to_input_json]`

## JSON Input Format

```json
{
    "challenge_info": {
        "challenge_id": "round_1b_002",
        "test_case_name": "travel_planner",
        "description": "France Travel"
    },
    "documents": [
        {
            "filename": "South of France - Cities.pdf",
            "title": "South of France - Cities"
        }
    ],
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip of 4 days for a group of 10 college friends."
    }
}
```

## Usage Examples

### Default Input File
```bash
python main.py
```
Uses `./input/challenge1.json`

### Custom Input File
```bash
python main.py "Challenge_1b/Collection 1/challenge1b_input.json"
```

### Process All Challenge 1B Collections
```bash
process_challenge1b.bat
```

## Folder Structure
```
Round 1B/
├── PDFs/                    # PDF files to process
├── input/                   # JSON input files
├── output/                  # Generated results
├── Challenge_1b/            # Challenge collections
│   ├── Collection 1/
│   ├── Collection 2/
│   └── Collection 3/
├── main.py                  # Main processing script
└── process_challenge1b.bat  # Batch processing script
```

## Output
Results are saved to the `output/` folder with timestamped filenames based on the persona and job description.