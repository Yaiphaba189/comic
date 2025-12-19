# AI Comic Generator

A web application that generates comics from stories using AI.

## Features

- **Story Parsing**: Natural Language Processing to understand story structure.
- **Emotion Analysis**: Text-based emotion detection for character expressions.
- **Image Generation**: Stable Diffusion for creating comic panels.
- **Dynamic Layout**: Automatic comic page layout generation.

## Tech Stack

### Backend

- **Framework**: FastAPI
- **ML/AI**: PyTorch, Diffusers, Scikit-learn, OpenCV
- **Language**: Python

### Frontend

- **Framework**: Next.js
- **Styling**: TailwindCSS
- **Language**: TypeScript

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js & npm

### Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser.
