import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    
    # Carregar o modelo treinado
    print(f"Carregando modelo de: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Converter para TensorFlow Lite com quantização Dynamic Range
    print("Convertendo para TensorFlow Lite com otimização...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Aplicar Dynamic Range Quantization para otimização
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Converter
    tflite_model = converter.convert()
    
    # Salvar o modelo otimizado
    tflite_path = os.path.join(script_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Modelo otimizado salvo em: {tflite_path}")
    
    # Mostrar informações de tamanho
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    
    print(f"\nTamanho do modelo original (model.h5): {model_size:.2f} MB")
    print(f"Tamanho do modelo otimizado (model.tflite): {tflite_size:.2f} MB")
    print(f"Redução: {(1 - tflite_size / model_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
