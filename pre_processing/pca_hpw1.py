import numpy as np
import os
import glob
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging

# --- Configuração Centralizada ---
CONFIG = {
    "input_dir": Path('./../../hyperview1/normalized_npy/'),
    "output_dir": Path("./../../hyperview1/pca/"),
    "total_pca_components": 32,
    "num_band_groups": 4,
    "patch_dims": (150, 10, 10),  # (C, H, W)
}

# Configuração de logging para melhor feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_patch(filepath, expected_dims):
    # ... (esta função permanece a mesma)
    try:
        data = np.load(filepath)
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        
        if data.shape != expected_dims:
            logging.warning(f"Arquivo {filepath} com dimensões inesperadas {data.shape}. Ignorando.")
            return None, None

        data_hwc = np.moveaxis(data, 0, -1)
        h, w, c = data_hwc.shape
        
        pixel_matrix = data_hwc.reshape(-1, c)
        return pixel_matrix, (h, w)
        
    except Exception as e:
        logging.error(f"Erro ao processar o arquivo {filepath}: {e}")
        return None, None


def split_data(data_list, group=4):
    output_data = data_list
    step = group // 2
    for i in range(step):
        split_data = []
        for data in output_data:
            n, c = data.shape
            data_s1 = data[:, :c // 2]
            data_s2 = data[:, c // 2:]
            split_data.append(data_s1)
            split_data.append(data_s2)
        output_data = split_data
    return output_data


def train_gwpca_models(files, config):
    logging.info("Iniciando a fase de treinamento dos modelos Group-wise Incremental PCA...")
    
    num_groups = config["num_band_groups"]
    components_per_group = config["total_pca_components"] // num_groups
    
    ipca_models = [IncrementalPCA(n_components=components_per_group, whiten=True) for _ in range(num_groups)]
    
    for file_path in tqdm(files, desc="Treinando modelos IPCA"):
        pixel_matrix, _ = load_and_preprocess_patch(file_path, config["patch_dims"])

        if pixel_matrix is not None:
            grouped_data = split_data([pixel_matrix], group=num_groups)
            for i, group_data in enumerate(grouped_data):
                ipca_models[i].partial_fit(group_data)



    logging.info("Treinamento dos modelos concluído.")
    return ipca_models


def transform_and_save_data(files, trained_models, config):
    # ... (esta função permanece a mesma, mas se beneficiará de modelos mais estáveis)
    logging.info(f"Iniciando a fase de transformação para {len(files)} arquivos...")
    
    input_base_dir = config["input_dir"]
    output_base_dir = config["output_dir"]

    for file_path in tqdm(files, desc="Transformando e salvando patches"):
        pixel_matrix, original_shape = load_and_preprocess_patch(file_path, config["patch_dims"])
        
        if pixel_matrix is not None:
            band_groups = split_data([pixel_matrix], config["num_band_groups"])
            transformed_groups = []
            
            for group_idx, group_data in enumerate(band_groups):
                # Se o grupo de dados original tiver variância zero, sua transformação também
                # terá variância zero. Podemos criar um array de zeros com o formato correto.
                if np.std(group_data) > 1e-9:
                    transformed_data = trained_models[group_idx].transform(group_data)
                else:
                    n_samples = group_data.shape[0]
                    n_components = trained_models[group_idx].n_components_
                    transformed_data = np.zeros((n_samples, n_components))

                transformed_groups.append(transformed_data)
            
            pca_result = np.concatenate(transformed_groups, axis=1)
            h, w = original_shape
            output_image = pca_result.reshape(h, w, -1)
            
            relative_path = file_path.relative_to(input_base_dir)
            output_path = output_base_dir / relative_path
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, output_image)


def main():
    # ... (a função main permanece a mesma)
    if CONFIG["total_pca_components"] % CONFIG["num_band_groups"] != 0:
        raise ValueError("O número total de componentes deve ser divisível pelo número de grupos.")

    input_dir = CONFIG["input_dir"]
    train_dir = input_dir / "train"
    test_dir = input_dir / "test"

    logging.info(f"Procurando arquivos de TREINO em: {train_dir}")
    train_files = list(train_dir.rglob('*.npy'))
    if not train_files:
        logging.error("Nenhum arquivo de treino .npy encontrado. Verifique o diretório.")
        return
    logging.info(f"Encontrados {len(train_files)} arquivos de treino.")

    logging.info(f"Procurando arquivos de TESTE em: {test_dir}")
    test_files = list(test_dir.rglob('*.npy'))
    if not test_files:
        logging.warning("Nenhum arquivo de teste .npy encontrado.")
    else:
        logging.info(f"Encontrados {len(test_files)} arquivos de teste.")

    trained_models = train_gwpca_models(train_files + test_files, CONFIG)
    
    transform_and_save_data(train_files, trained_models, CONFIG)
    
    if test_files:
        transform_and_save_data(test_files, trained_models, CONFIG)
    
    logging.info(f"\nProcesso concluído com sucesso!")
    logging.info(f"Dados transformados salvos em: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()