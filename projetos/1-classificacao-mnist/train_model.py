import os
import tensorflow as tf
# Use keras via the tensorflow namespace to avoid import-from-source issues
# (some linters/editors cannot resolve `from tensorflow.keras import ...`)
keras = tf.keras
layers = keras.layers

# Forçar execução em CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

def main():
    print("Carregando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalizar para [0, 1] e expandir para (28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    print(f"Shape dos dados: {x_train.shape}")
    
    # Construir a CNN com 4 blocos convolucionais
    print("\nConstruindo modelo CNN...")
    model = keras.Sequential([
        # Bloco 1: Conv2D + BatchNorm + MaxPool
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", 
                      input_shape=(28, 28, 1), padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 2: Conv2D + BatchNorm + MaxPool
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 3: Conv2D + BatchNorm + MaxPool
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 4: Conv2D + BatchNorm
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        
        # Flatten e Dropout
        layers.Flatten(),
        layers.Dropout(0.5),
        
        # Camada de saída
        layers.Dense(10, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # Treinar com EarlyStopping, validação explícita e checkpoint do melhor modelo
    print("\nTreinando o modelo com CPU...")

    # Parâmetros ajustáveis via variáveis de ambiente para execuções curtas/CI
    epochs = int(os.environ.get("EPOCHS", 15))
    batch_size = int(os.environ.get("BATCH_SIZE", 128))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Avaliar no conjunto de teste
    print("\nAvaliando no conjunto de teste...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Acurácia de validação final: {test_accuracy:.4f}")
    print(f"Perda de validação final: {test_loss:.4f}")

    # O melhor modelo já foi salvo por ModelCheckpoint (model.h5)
    if os.path.exists(model_path):
        print(f"\nModelo salvo em: {model_path}")
    else:
        # Fallback: salvar o modelo atual
        model.save('model.h5', save_format='tf')
        print(f"\nModelo salvo em: {model_path} (fallback)")


if __name__ == "__main__":
    import numpy as np
    main()
