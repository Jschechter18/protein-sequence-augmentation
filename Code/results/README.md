# Folder Organization

### autoencoder/

Contains autoencoder training results (Token accuracy)

Solubility:
v1:

- Teacher forcing:
  - Test Loss: 2.3473, Test Accuracy: 0.2567
- Autoregressive:
  - Test Loss: 3.2196, Test Accuracy: 0.1176

v2:

- Teacher forcing:
  - Test Loss: 2.1800, Test Accuracy: 0.2979
- Autoregressive:
  - Test Loss: 3.0678, Test Accuracy: 0.1311

v3:

- Teacher forcing:
  - Test Loss: 2.5239, Test Accuracy: 0.2039
- Autoregressive:
  - Test Loss: 2.9446, Test Accuracy: 0.1005

v4:

- Teacher forcing:
  - Test Loss: 1.6853, Test Accuracy: 0.3994
- Autoregressive:
  - Test Loss: 3.4033, Test Accuracy: 0.1439

v5: (official benchmark)

- history json: protein-sequence-augmentation/Code/results/autoencoder/solubility/v5/solubility_ae_history.json

- Teacher forcing:
  - Test Loss: 1.1311, Test Accuracy: 0.5819
- Autoregressive:
  - Test Loss: 3.3443, Test Accuracy: 0.2538

### esm2/

Contains ESM-2 and 1DCNN results

### figures/

### tables/

Final benchmark tables
