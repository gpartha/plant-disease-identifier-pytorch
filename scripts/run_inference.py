"""Inference script for plant disease detection."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from src.inference.predictor import PlantDiseasePredictor
from config.config import Config

def main():
    parser = argparse.ArgumentParser(description='Run inference on plant disease images')
    parser.add_argument('--model_path', type=str, 
                       default='data/models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image for prediction')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Initialize predictor
    config = Config()
    predictor = PlantDiseasePredictor(args.model_path, config)
    
    # Make prediction
    results = predictor.predict_image_path(args.image_path, args.top_k)
    
    if results:
        print(f"\nPredictions for {args.image_path}:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['disease']}: {result['confidence']:.2f}%")
    else:
        print("Prediction failed!")

if __name__ == "__main__":
    main()
