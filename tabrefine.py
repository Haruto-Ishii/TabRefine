import numpy as np
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, LabelEncoder, OrdinalEncoder, MinMaxScaler
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tabm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from typing import Optional
import rtdl_num_embeddings


dataset_name = "UNSW-NB15" # "CIC-IDS2018", "UNSW-NB15", or "CICIoT2023"
baseline_method = "STaSy" # "STaSy", "TabDDPM", or "TabSyn" 
AE_HIDDEN_DIM = 128
AE_LR = 0.001
AE_EPOCHS = 50
AE_BATCH_SIZE = 8192
GEN_HIDDEN_DIM = 2048
DIS_D_MODEL = 128
DIS_N_HEADS = 4
DIS_N_LAYERS = 2
DIS_DIM_FEEDFORWARD_FACTOR = 4
DIS_ACTIVATION_FUNCTION = 'relu'
DIS_EPOCHS = 400
GEN_EPOCHS = 2000
GAN_LR = 0.0002
GAN_BATCH_SIZE = 4096
INITIAL_ALPHA = 100

real_train_num = np.load(f"./real_data/{dataset_name}/train_data_num.npy").astype(np.float64)
real_train_cat = np.load(f"./real_data/{dataset_name}/train_data_cat.npy", allow_pickle=True).astype(np.int32)
real_train_target = np.load(f"./real_data/{dataset_name}/train_data_target.npy").astype('int')

synth_data_num = np.load(f"./synth_data/{dataset_name}/{baseline_method}/Syn_num.npy").astype(np.float64)
synth_data_cat = np.load(f"./synth_data/{dataset_name}/{baseline_method}/Syn_cat.npy", allow_pickle=True).astype(np.int32)
synth_data_target = np.load(f"./synth_data/{dataset_name}/{baseline_method}/Syn_target.npy").astype('int')


def preprocess_data(real_train_num, real_train_cat, real_train_target, synth_data_num, synth_data_cat, synth_data_target):
    num_numerical_features = real_train_num.shape[1]
    num_classes = len(np.unique(real_train_target))
    
    label_map = LabelEncoder()
    label_map.fit(real_train_target)
    real_target_mapped = label_map.transform(real_train_target)
    syn_target_mapped = label_map.transform(synth_data_target)
    
    quantile_transformer = QuantileTransformer(output_distribution="normal", random_state=42)
    real_num_scaled = quantile_transformer.fit_transform(real_train_num)
    syn_num_scaled = quantile_transformer.transform(synth_data_num)
    
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    onehot_encoder.fit(real_train_cat)
    real_cat_scaled = onehot_encoder.transform(real_train_cat)
    syn_cat_scaled = onehot_encoder.transform(synth_data_cat)
    
    real_features = np.hstack([real_num_scaled, real_cat_scaled])
    syn_features = np.hstack([syn_num_scaled, syn_cat_scaled])
    
    return quantile_transformer, onehot_encoder, real_features, real_target_mapped, syn_features, syn_target_mapped, num_numerical_features, num_classes, label_map

quantile_transformer, onehot_encoder, real_features, real_label, syn_features, syn_label, num_numerical_features, num_classes, label_map =\
    preprocess_data(real_train_num, real_train_cat, real_train_target, synth_data_num, synth_data_cat, synth_data_target)

class CategoryAE(nn.Module):
    def __init__(self, one_hot_dim, original_cat_dim, hidden_dim=AE_HIDDEN_DIM):
        super(CategoryAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(one_hot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, original_cat_dim),
            nn.BatchNorm1d(original_cat_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(original_cat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, one_hot_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x_one_hot):
        return self.encoder(x_one_hot)
    
    def decode(self, z_cat):
        return self.decoder(z_cat)
    
    def forward(self, x_one_hot):
        z = self.encode(x_one_hot)
        recon_x = self.decode(z)
        return recon_x
    
class CategoryAETrainer:
    def __init__(self, model, lr=AE_LR, batch_size=AE_BATCH_SIZE):
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.device = next(model.parameters()).device
    def train(self, data, epochs=AE_EPOCHS):
        self.model.train()
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print("--- Starting CategoryAE Training ---")
        for epoch in range(epochs):
            total_loss = 0
            for batch_tup in loader:
                batch = batch_tup[0].to(self.device)
                self.optimizer.zero_grad()
                recon_batch = self.model(batch)
                loss = self.loss_fn(recon_batch, batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} / {epochs}, Loss: {total_loss/len(loader.dataset)}")
        print("--- CategoryAE Training Completed ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"We use {device}")
one_hot_dim = real_features.shape[1] - num_numerical_features
original_cat_dim = real_train_cat.shape[1]

cat_ae = CategoryAE(one_hot_dim, original_cat_dim).to(device)
cat_ae_trainer = CategoryAETrainer(cat_ae)

combined_cat_data = torch.tensor(np.vstack([real_features[:, num_numerical_features:], syn_features[:, num_numerical_features:]]), dtype=torch.float32).to(device)
cat_ae_trainer.train(combined_cat_data)

cat_ae.eval()
with torch.no_grad():
    real_cat_encoded = cat_ae.encode(combined_cat_data[:len(real_label)])
    syn_cat_encoded = cat_ae.encode(combined_cat_data[len(real_label):])
    
real_combined = torch.cat([torch.tensor(real_features[:, :num_numerical_features], dtype=torch.float32).to(device), real_cat_encoded], dim=1)
syn_combined = torch.cat([torch.tensor(syn_features[:, :num_numerical_features], dtype=torch.float32).to(device), syn_cat_encoded], dim=1)

class MLPGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=GEN_HIDDEN_DIM):
        super(MLPGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
class TransformerDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=DIS_D_MODEL, n_heads=DIS_N_HEADS, n_layers=DIS_N_LAYERS, dim_feedforward_factor=DIS_DIM_FEEDFORWARD_FACTOR, activation_function=DIS_ACTIVATION_FUNCTION):
        super(TransformerDiscriminator, self).__init__()

        self.feature_embed = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * dim_feedforward_factor,
            batch_first=True,
            activation=activation_function
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )

        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x_embedded = self.feature_embed(x)
        
        x_seq = x_embedded.unsqueeze(1)
        
        transformer_out = self.transformer_encoder(x_seq)
        
        h = transformer_out.squeeze(1)

        delta = self.output_layer(h)
        
        return delta

class GANTrainer:
    def __init__(self, generator, discriminator, num_classes, d_epochs=DIS_EPOCHS, g_epochs=GEN_EPOCHS, lr=GAN_LR, initial_alpha=INITIAL_ALPHA):
        
        self.generator = generator
        self.discriminator = discriminator
        self.d_epochs = d_epochs
        self.g_epochs = g_epochs
        self.initial_alpha = initial_alpha
        self.device = next(generator.parameters()).device
        self.num_classes = num_classes
        
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr)
        self.optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.regularization_loss = nn.MSELoss()
        
    def _create_multilabel_targets(self, raw_labels, is_real):

        batch_size = raw_labels.size(0)
        targets = torch.zeros(batch_size, self.num_classes, device=self.device)
        
        if is_real:
            targets.scatter_(1, raw_labels.unsqueeze(1), 1)
        
        return targets

    def train_discriminator(self, real_discriminator_input, syn_discriminator_input, real_raw_labels, syn_raw_labels):
        print("--- Starting Discriminator Training ---")
        
        d_labels_real = self._create_multilabel_targets(real_raw_labels, is_real=True)
        d_labels_syn = self._create_multilabel_targets(syn_raw_labels, is_real=False)
        
        d_dataset = TensorDataset(
            torch.cat([real_discriminator_input, syn_discriminator_input], dim=0),
            torch.cat([d_labels_real, d_labels_syn], dim=0)
        )
        d_loader = DataLoader(d_dataset, batch_size=GAN_BATCH_SIZE, shuffle=True)
        
        self.discriminator.train()
        for epoch in range(self.d_epochs):
            total_loss = 0
            for data, labels in d_loader:
                self.optimizer_d.zero_grad()
                preds = self.discriminator(data)
                loss = self.adversarial_loss(preds, labels)
                total_loss += loss.item()
                loss.backward()
                self.optimizer_d.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1} / {self.d_epochs}, Loss {total_loss / len(d_loader)}")
        print("--- Discriminator Training Completed ---")
        
    def train_generator(self, syn_generator_input, syn_raw_labels):
        print("--- Starting Generator Training ---")
        
        label_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        all_possible_labels = np.arange(self.num_classes).reshape(-1, 1)
        label_encoder.fit(all_possible_labels)
        
        g_dataset = TensorDataset(syn_generator_input, syn_raw_labels)
        g_loader = DataLoader(g_dataset, batch_size=GAN_BATCH_SIZE, shuffle=True)
        
        self.discriminator.eval()
        alpha = self.initial_alpha
        patience = 0

        self.generator.train()
        for epoch in range(self.g_epochs):
            total_loss, total_loss_adv, total_loss_reg = 0, 0, 0
            for syn_data_batch, syn_label_batch in g_loader:
                syn_data_batch = syn_data_batch.to(self.device)
                syn_label_batch = syn_label_batch.to(self.device)
                self.optimizer_g.zero_grad()
                
                syn_label_onehot = torch.tensor(
                    label_encoder.transform(syn_label_batch.cpu().numpy().reshape(-1, 1)),
                    dtype=torch.float32,
                    device=self.device
                )
                generator_input = torch.cat([syn_data_batch, syn_label_onehot], dim=1)

                delta = self.generator(generator_input)
                refined_data = syn_data_batch + delta
                preds = self.discriminator(refined_data)
                
                target_labels = self._create_multilabel_targets(syn_label_batch, is_real=True)
                loss_adv = self.adversarial_loss(preds, target_labels)
                loss_reg = self.regularization_loss(delta, torch.zeros_like(delta))
                loss_g = loss_adv + alpha * loss_reg
                total_loss += loss_g.item()
                total_loss_adv += loss_adv.item()
                total_loss_reg += loss_reg.item()
                
                loss_g.backward()
                self.optimizer_g.step()

            if (epoch + 1) % 10 == 0:
                self.generator.eval()
                with torch.no_grad():
                    successful_foolings = 0
                    total_samples = 0
                    for eval_batch, eval_label_batch in g_loader:
                        eval_batch = eval_batch.to(self.device)
                        eval_label_batch = eval_label_batch.to(self.device)
                        
                        eval_label_onehot = torch.tensor(
                            label_encoder.transform(eval_label_batch.cpu().numpy().reshape(-1,1)),
                            dtype=torch.float32,
                            device=self.device
                        )
                        eval_generator_input = torch.cat([eval_batch, eval_label_onehot], dim=1)
                        
                        delta = self.generator(eval_generator_input)
                        refined = eval_batch + delta
                        eval_preds = self.discriminator(refined)
                        
                        eval_probs = torch.sigmoid(eval_preds)
                        true_class_probs = torch.gather(eval_probs, 1, eval_label_batch.unsqueeze(1))
                        
                        successful_foolings_batch = (true_class_probs > 0.5).sum().item()
                        
                        successful_foolings += successful_foolings_batch
                        
                        total_samples += len(eval_batch)

                accuracy = successful_foolings / total_samples
                print(f"Epoch {epoch+1}/{self.g_epochs}, Avg Fooling Probability: {accuracy:.4f}, Current Alpha: {alpha:.2f}, Total Loss: {total_loss:.4f} Loss adv: {total_loss_adv:.4f}, Loss reg: {total_loss_reg:.4f}")

                if epoch > 500 and accuracy < 0.8:
                    patience += 1
                    if patience >= 5:
                        alpha = max(alpha * 0.7, 1.0)
                        print(f"Accuracy is low, reducing alpha to {alpha:.2f}")
                        patience = 0
                elif accuracy > 0.95:
                    alpha *= 1.1
                    print(f"Accuracy is high, increasing alpha to {alpha:.2f}")
                    patience = 0
                else:
                    patience = 0
                self.generator.train()
    
    def train(self, real_features, syn_features, real_raw_labels, syn_raw_labels):
        real_raw_labels = torch.from_numpy(real_raw_labels).to(device=self.device, dtype=torch.long)
        syn_raw_labels = torch.from_numpy(syn_raw_labels).to(device=self.device, dtype=torch.long)
            
        self.train_discriminator(real_features, syn_features, real_raw_labels, syn_raw_labels)
        
        self.train_generator(syn_features, syn_raw_labels)
        return self.generator.eval()

geenrator_input_dim = real_combined.shape[1] + num_classes
generator_output_dim = real_combined.shape[1]
discriminator_input_dim = real_combined.shape[1]

generator = MLPGenerator(geenrator_input_dim, generator_output_dim).to(device)
discriminator = TransformerDiscriminator(discriminator_input_dim, num_classes).to(device)

gan_trainer = GANTrainer(generator, discriminator, num_classes)

trained_generator = gan_trainer.train(real_combined, syn_combined, real_label, syn_label)

trained_generator.eval()
with torch.no_grad():
    label_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    all_possible_labels = np.arange(num_classes).reshape(-1, 1)
    label_encoder.fit(all_possible_labels)
    syn_label_onehot = label_encoder.transform(syn_label.reshape(-1, 1))
    syn_label_onehot = torch.tensor(syn_label_onehot, dtype=torch.float32, device=device)
    generator_input = torch.hstack([syn_combined, syn_label_onehot])
    delta = trained_generator(generator_input)
    refined_syn_combined = syn_combined + delta
    
refined_syn_num = refined_syn_combined[:, :num_numerical_features].cpu().numpy()
refined_syn_cat = refined_syn_combined[:, num_numerical_features:]

with torch.no_grad():
    refined_syn_cat_onehot = cat_ae.decode(refined_syn_cat).cpu().numpy()
    
final_syn_num = quantile_transformer.inverse_transform(refined_syn_num)
final_syn_cat = onehot_encoder.inverse_transform(refined_syn_cat_onehot)
final_syn_label = label_map.inverse_transform(syn_label)

synth_save_dir = f'./refined_data/{dataset_name}/{baseline_method}'
os.makedirs(os.path.abspath(synth_save_dir), exist_ok=True)
np.save(f'{synth_save_dir}/Syn_num.npy', final_syn_num)
np.save(f'{synth_save_dir}/Syn_cat.npy', final_syn_cat)
np.save(f'{synth_save_dir}/Syn_target.npy', final_syn_label)