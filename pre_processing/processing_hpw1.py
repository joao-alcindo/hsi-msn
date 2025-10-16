import pandas
import numpy as np
import glob
import os
from pathlib import Path
from tqdm import tqdm

# --- Configuração Centralizada ---
config = {
    "base_dir": Path('./../../hyperview1/'),
    "patch_size": 10,
    "n_bands": 150,
}
config["train_dir"] = config["base_dir"] / "train_data/train_data/train_data"
config["test_dir"] = config["base_dir"] / "test_data"
config["output_dir"] = config["base_dir"] / "normalized_npy"


# --- Bloco de Cálculo de Estatísticas (Método Original do Usuário) ---
def compute_stats_original(train_files, test_files, n_bands):
    """
    Calcula as estatísticas carregando todos os dados para a memória,
    conforme o método do script original.
    
    AVISO: Pode consumir muita RAM para datasets grandes.
    """
    print("Carregando todos os dados de treino e teste para a memória...")
    data_train = [np.load(f)['data'] for f in tqdm(train_files, desc="Carregando treino")]
    data_test = [np.load(f)['data'] for f in tqdm(test_files, desc="Carregando teste")]

    print("Concatenando dados...")
    flat_train = np.concatenate([img.reshape(n_bands, -1) for img in data_train], axis=1)
    flat_test = np.concatenate([img.reshape(n_bands, -1) for img in data_test], axis=1)
    
    flat = np.concatenate([flat_train, flat_test], axis=1)

    print("Calculando estatísticas (Média, Desvio Padrão, Percentis)...")
    p1 = np.percentile(flat, 1, axis=1)
    p99 = np.percentile(flat, 99, axis=1)

    flat = np.clip(flat, p1[:, None], p99[:, None])

    mean = np.mean(flat, axis=1)
    std = np.std(flat, axis=1)

    # Liberar a memória dos arrays gigantes, como no original
    del flat_train, flat_test, data_train, data_test, flat
    print("Cálculo de estatísticas concluído.")
    
    stats = {'mean': mean, 'std': std, 'p1': p1, 'p99': p99}
    return stats


def create_patches_vectorized(image, patch_size):
    """
    [MELHORIA MANTIDA] Cria patches de forma vetorizada com NumPy, muito mais rápido.
    """
    C, H, W = image.shape
    
    h_new = H - H % patch_size
    w_new = W - W % patch_size
    cropped_image = image[:, :h_new, :w_new]
    
    patches = cropped_image.reshape(C, h_new // patch_size, patch_size, w_new // patch_size, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    
    return patches



def process_and_save_files(files, output_dir, stats, patch_size):
    """
    Função unificada para processar arquivos. A lógica de salvamento
    foi REVERTIDA para o método original (um arquivo por patch).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mean = stats['mean'][:, None, None]
    std = stats['std'][:, None, None]
    p1 = stats['p1'][:, None, None]
    p99 = stats['p99'][:, None, None]

    for f in tqdm(files, desc=f"Processando e salvando patches em {output_dir.name}"):
        data = np.load(f)['data']

        clipped_data = np.clip(data, p1, p99)
        
        norm_data = (clipped_data - mean) / std

        patches = create_patches_vectorized(norm_data, patch_size)

        # --- [REVERTIDO] Lógica de salvamento original (um arquivo por patch) ---
        # Salva cada patch como um arquivo .npy individual.
        base_filename = Path(f).stem
        for idx in range(patches.shape[0]):
            # Formata o nome do arquivo como: nome_original_patch001.npy
            out_filename = f"{base_filename}_patch{idx:03d}.npy"
            out_path = output_dir / out_filename
            np.save(out_path, patches[idx])
        
        


def main():
    """Função principal que orquestra todo o processo."""
    
    train_files = sorted(glob.glob(str(config["train_dir"] / '*.npz')))
    test_files = sorted(glob.glob(str(config["test_dir"] / '*.npz')))
    
    if not (train_files and test_files):
        print("Nenhum arquivo .npz encontrado em treino ou teste. Verifique os caminhos.")
        return

    # 1. Calcular estatísticas globais com o método original
    stats = compute_stats_original(train_files, test_files, config['n_bands'])

    # 2. Definir diretórios de saída
    train_output_dir = config["output_dir"] / "train"
    test_output_dir = config["output_dir"] / "test"

    # 3. Processar e salvar os patches de treino e teste
    process_and_save_files(train_files, train_output_dir, stats, config['patch_size'])
    process_and_save_files(test_files, test_output_dir, stats, config['patch_size'])

    print("\nProcesso concluído com sucesso!")
    print(f"Patches salvos em: {config['output_dir']}")


if __name__ == '__main__':
    main()