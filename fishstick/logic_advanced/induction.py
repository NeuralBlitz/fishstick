import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ProgramOp(Enum):
    APPEND = "append"
    MAP = "map"
    FILTER = "filter"
    REDUCE = "reduce"
    IF = "if"
    WHILE = "while"
    INDEX = "index"
    SLICE = "slice"


@dataclass
class Program:
    ops: List[ProgramOp]
    args: List[Any]
    description: str = ""

    def __repr__(self) -> str:
        parts = []
        for op, arg in zip(self.ops, self.args):
            if arg is not None:
                parts.append(f"{op.value}({arg})")
            else:
                parts.append(op.value)
        return " → ".join(parts)


@dataclass
class ProgramEmbedding:
    program_id: int
    embedding: torch.Tensor
    program: Optional[Program] = None


class NeuralProgramInducer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_ops: int = 8,
        max_program_length: int = 20,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_ops = num_ops
        self.max_program_length = max_program_length
        self.embedding_dim = embedding_dim

        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.program_decoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.op_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_ops),
        )

        self.arg_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

        self.program_embedding = nn.Embedding(1000, embedding_dim)

        self.memory = nn.LSTMCell(hidden_dim, hidden_dim)

    def forward(
        self,
        input_examples: torch.Tensor,
        output_examples: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = input_examples.size(0)

        input_encoded = self.input_encoder(input_examples)

        h = torch.zeros(batch_size, self.hidden_dim, device=input_examples.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=input_examples.device)

        program_logits = []
        program_args = []

        for step in range(self.max_program_length):
            h, c = self.memory(input_encoded, (h, c))

            op_logits = self.op_predictor(h)
            program_logits.append(op_logits)

            arg_logits = self.arg_predictor(h)
            program_args.append(arg_logits)

        all_logits = torch.stack(program_logits, dim=1)
        all_args = torch.stack(program_args, dim=1)

        return {
            "op_logits": all_logits,
            "arg_logits": all_args,
            "hidden_state": h,
        }

    def induce_program(
        self,
        input_output_pairs: List[Tuple[Any, Any]],
    ) -> Program:
        ops = []
        args = []

        for _ in range(min(5, self.max_program_length)):
            op_idx = np.random.randint(0, self.num_ops)
            arg = np.random.randn(10)
            ops.append(ProgramOp(op_idx))
            args.append(arg.tolist())

        return Program(ops=ops, args=args, description="Induced program")

    def execute_program(
        self,
        program: Program,
        inputs: List[Any],
    ) -> List[Any]:
        results = inputs.copy()

        for op, arg in zip(program.ops, program.args):
            if op == ProgramOp.APPEND:
                results.append(arg[0] if arg else None)
            elif op == ProgramOp.MAP:
                results = [x * 2 for x in results]
            elif op == ProgramOp.FILTER:
                results = [x for x in results if x > 0]
            elif op == ProgramOp.REDUCE:
                if results:
                    results = [sum(results)]
            elif op == ProgramOp.INDEX:
                if results and isinstance(arg[0], int):
                    results = [results[arg[0] % len(results)]]
            elif op == ProgramOp.SLICE:
                results = results[:3]

        return results

    def learn_from_examples(
        self,
        input_examples: List[torch.Tensor],
        output_examples: List[torch.Tensor],
    ) -> float:
        input_tensor = torch.stack(input_examples)
        output_tensor = torch.stack(output_examples)

        result = self.forward(input_tensor, output_tensor)

        target_ops = torch.randint(0, self.num_ops, (input_tensor.size(0), 5))
        target_args = torch.randn(input_tensor.size(0), 5, 10)

        op_loss = nn.functional.cross_entropy(
            result["op_logits"].view(-1, self.num_ops),
            target_ops.view(-1),
        )
        arg_loss = nn.functional.mse_loss(
            result["arg_logits"][:, :5, :],
            target_args,
        )

        loss = op_loss + arg_loss
        loss.backward()

        return loss.item()


class ProgramInduction(nn.Module):
    def __init__(
        self,
        domain_dim: int,
        hidden_dim: int = 256,
        num_primitives: int = 20,
    ):
        super().__init__()
        self.domain_dim = domain_dim
        self.hidden_dim = hidden_dim

        self.domain_encoder = nn.Sequential(
            nn.Linear(domain_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.primitive_library = nn.Embedding(num_primitives, hidden_dim)

        self.search_policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.composer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        domain_spec: torch.Tensor,
        candidates: List[torch.Tensor],
    ) -> torch.Tensor:
        domain_encoded = self.domain_encoder(domain_spec)

        scores = []
        for candidate in candidates:
            combined = torch.cat([domain_encoded, candidate], dim=-1)
            score = self.search_policy(combined)
            scores.append(score)

        return torch.stack(scores).squeeze(-1)

    def synthesize(
        self,
        specification: Dict[str, Any],
        primitives: List[str],
    ) -> Program:
        ops = []
        args = []

        for _ in range(np.random.randint(2, 6)):
            op_choice = np.random.choice(list(ProgramOp))
            ops.append(op_choice)
            args.append(np.random.randn(5).tolist())

        return Program(
            ops=ops,
            args=args,
            description=f"Synthesized program for {specification.get('task', 'unknown')}",
        )

    def verify_program(
        self,
        program: Program,
        test_cases: List[Tuple[Any, Any]],
    ) -> float:
        if not test_cases:
            return 0.0

        passed = 0
        for input_val, expected_output in test_cases:
            try:
                result = self._execute_on_input(program, input_val)
                if result == expected_output:
                    passed += 1
            except:
                pass

        return passed / len(test_cases)

    def _execute_on_input(self, program: Program, input_val: Any) -> Any:
        return input_val


class ProgramSynthesizer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        max_depth: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=3,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=3,
        )

        self.program_head = nn.Linear(embedding_dim, vocab_size)

        self.constraint_check = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        io_examples: torch.Tensor,
        partial_program: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded = self.token_embedding(io_examples)

        encoded = self.encoder(embedded)

        if partial_program is not None:
            partial_emb = self.token_embedding(partial_program)
            decoded = self.decoder(partial_emb, encoded)
        else:
            decoded = encoded

        program_logits = self.program_head(decoded)

        return {
            "program_logits": program_logits,
            "encoded": encoded,
            "decoded": decoded,
        }

    def generate_program(
        self,
        io_pairs: List[Tuple[List[int], List[int]]],
    ) -> List[int]:
        io_tensor = torch.tensor(
            [p[0] + p[1] for p in io_pairs],
            dtype=torch.long,
        )

        with torch.no_grad():
            result = self.forward(io_tensor)
            program_ids = result["program_logits"].argmax(dim=-1)

        return program_ids[0].tolist()

    def npi_forward(
        self,
        program_context: torch.Tensor,
        input_data: torch.Tensor,
        arg_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_emb = self.token_embedding(program_context)
        input_emb = self.token_embedding(input_data)

        combined = torch.cat([context_emb, input_emb], dim=-1)

        encoded = self.encoder(combined.unsqueeze(0))

        program_logits = self.program_head(encoded)

        return program_logits.squeeze(0), arg_stack

    def infer_arguments(
        self,
        program: Program,
        input_example: Any,
    ) -> Dict[str, Any]:
        inferred_args = {}

        for i, (op, arg) in enumerate(zip(program.ops, program.args)):
            if op == ProgramOp.INDEX:
                inferred_args[f"index_{i}"] = 0
            elif op == ProgramOp.SLICE:
                inferred_args[f"slice_{i}"] = (0, 3)
            else:
                inferred_args[f"arg_{i}"] = arg[:3] if arg else []

        return inferred_args
