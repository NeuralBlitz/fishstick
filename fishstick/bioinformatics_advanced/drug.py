import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class Molecule:
    smiles: str
    atoms: torch.Tensor
    bonds: torch.Tensor
    embeddings: Optional[torch.Tensor] = None


class MolecularDocking(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int = 62,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.interaction_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.docking_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ligand_atoms: torch.Tensor,
        ligand_edges: torch.Tensor,
        receptor_atoms: torch.Tensor,
        receptor_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ligand_emb = self.atom_encoder(ligand_atoms)
        receptor_emb = self.atom_encoder(receptor_atoms)

        for layer in self.interaction_layers:
            combined = torch.cat([ligand_emb, receptor_emb], dim=1)
            combined = layer(combined)

            ligand_emb = combined[:, : ligand_emb.size(1)]
            receptor_emb = combined[:, ligand_emb.size(1) :]

        ligand_pooled = ligand_emb.mean(dim=1)
        receptor_pooled = receptor_emb.mean(dim=1)

        combined_features = torch.cat([ligand_pooled, receptor_pooled], dim=-1)

        docking_score = self.docking_scorer(combined_features)

        confidence = self.confidence_predictor(ligand_pooled)

        return docking_score, confidence

    def score_docking(
        self,
        ligand_smiles: str,
        receptor_pdb: str,
    ) -> Tuple[float, float]:
        ligand_atoms = torch.randn(1, 32, 62)
        ligand_edges = torch.randint(0, 2, (1, 32, 32, 12))
        receptor_atoms = torch.randn(1, 256, 62)
        receptor_edges = torch.randint(0, 2, (1, 256, 256, 12))

        with torch.no_grad():
            score, conf = self.forward(
                ligand_atoms,
                ligand_edges,
                receptor_atoms,
                receptor_edges,
            )

        return score.item(), conf.item()


class DrugTargetPredictor(nn.Module):
    def __init__(
        self,
        molecule_dim: int = 256,
        target_dim: int = 512,
        num_targets: int = 1000,
    ):
        super().__init__()

        self.molecule_encoder = nn.Sequential(
            nn.Linear(molecule_dim, molecule_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(molecule_dim, molecule_dim),
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, molecule_dim),
            nn.ReLU(),
            nn.Linear(molecule_dim, molecule_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(molecule_dim * 2, molecule_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(molecule_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        molecule_emb: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> torch.Tensor:
        mol_enc = self.molecule_encoder(molecule_emb)
        tgt_enc = self.target_encoder(target_emb)

        combined = torch.cat([mol_enc, tgt_enc], dim=-1)
        affinity = self.predictor(combined)

        return affinity

    def predict_affinity(
        self,
        molecule_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            affinity = self.forward(molecule_embedding, target_embedding)
        return affinity


class MoleculeGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        atom_vocab_size: int = 64,
        max_atoms: int = 100,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.atom_vocab_size = atom_vocab_size
        self.max_atoms = max_atoms

        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
        )

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
        )

        self.atom_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, atom_vocab_size),
        )

        self.bond_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)
        seq_len = target_length if target_length is not None else self.max_atoms

        h = self.latent_to_hidden(z)
        h = h.unsqueeze(0).repeat(3, 1, 1)

        atoms = torch.zeros(batch_size, seq_len, self.atom_vocab_size, device=z.device)
        atom_indices = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=z.device
        )

        hidden = h
        for t in range(seq_len):
            output, hidden = self.decoder(
                torch.zeros(batch_size, 1, self.latent_dim, device=z.device),
                hidden,
            )

            atom_logits = self.atom_predictor(output.squeeze(1))
            atoms[:, t] = atom_logits

            probs = F.softmax(atom_logits, dim=-1)
            atom_idx = torch.argmax(probs, dim=-1)
            atom_indices[:, t] = atom_idx

        edge_features = torch.zeros(batch_size, seq_len, seq_len, 4, device=z.device)
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    combined = torch.cat([atoms[:, i], atoms[:, j]], dim=-1)
                    edge_logits = self.bond_predictor(combined)
                    edge_features[:, i, j] = edge_logits

        mol_emb = atoms.mean(dim=1)
        properties = self.property_predictor(mol_emb)

        return atom_indices, edge_features

    def generate(
        self,
        num_molecules: int = 1,
        target_property: Optional[float] = None,
    ) -> List[Molecule]:
        z = torch.randn(
            num_molecules, self.latent_dim, device=next(self.parameters()).device
        )

        if target_property is not None:
            target_emb = torch.tensor([[target_property]], device=z.device).float()
            property_emb = F.relu(self.property_predictor.inverse()(target_emb))
            z = z + property_emb[:, : self.latent_dim]

        with torch.no_grad():
            atom_indices, edge_features = self.forward(z)

        smiles_list = []
        for i in range(num_molecules):
            smiles_list.append(f"MOL_{i}")

        return [
            Molecule(
                smiles=smiles,
                atoms=atom_indices[i],
                bonds=edge_features[i],
            )
            for i, smiles in enumerate(smiles_list)
        ]

    def generate_with_properties(
        self,
        target_solubility: Optional[float] = None,
        target_toxicity: Optional[float] = None,
    ) -> Molecule:
        z = torch.randn(1, self.latent_dim, device=next(self.parameters()).device)

        if target_solubility is not None:
            z = z + torch.randn_like(z) * target_solubility * 0.1

        atom_indices, edge_features = self.forward(z)

        return Molecule(
            smiles="GENERATED",
            atoms=atom_indices[0],
            bonds=edge_features[0],
        )


class GraphMoleculeEncoder(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int = 62,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)

        self.edge_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.atom_embedding(atom_features)

        batch_size, num_atoms, _ = x.shape

        for layer in self.edge_layers:
            x = layer(x)

        x = self.readout(x)

        pooled = x.mean(dim=1)

        return pooled


class VAEMoleculeGenerator(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int = 62,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        max_atoms: int = 100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms

        self.encoder = GraphMoleculeEncoder(
            atom_feature_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * max_atoms),
        )

        self.atom_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, atom_feature_dim),
        )

    def encode(
        self, atom_features: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(atom_features, edge_index)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        h = h.view(-1, self.max_atoms, self.hidden_dim)
        atom_features = self.atom_predictor(h)
        return atom_features

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(atom_features, edge_index)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar

    def generate_molecules(
        self,
        num_molecules: int = 10,
    ) -> List[Molecule]:
        z = torch.randn(
            num_molecules, self.latent_dim, device=next(self.parameters()).device
        )

        with torch.no_grad():
            atom_features = self.decode(z)

        return [
            Molecule(
                smiles=f"VAE_MOL_{i}",
                atoms=atom_features[i],
                bonds=torch.zeros(
                    self.max_atoms, self.max_atoms, 4, device=atom_features.device
                ),
            )
            for i in range(num_molecules)
        ]
