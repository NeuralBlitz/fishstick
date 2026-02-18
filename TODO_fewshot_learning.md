# Few-Shot Learning Module - TODO List

## Phase 1: Core Infrastructure
- [x] 1.1 Create `/home/runner/workspace/fishstick/fewshot_learning/` directory structure
- [x] 1.2 Create `__init__.py` with exports
- [x] 1.3 Create `config.py` - Configuration classes and defaults
- [x] 1.4 Create `types.py` - Type definitions and data structures

## Phase 2: MAML Implementations
- [x] 2.1 Create `maml.py` - Basic MAML implementation (extending existing)
- [x] 2.2 Create `fomaml.py` - First-order MAML (FOMAML)
- [x] 2.3 Create `anil.py` - Almost No Inner Loop (ANIL)
- [x] 2.4 Create `anil.py` - Bootstrapped Inner Loop (BOIL)
- [x] 2.5 Create `maml.py` - Meta-SGD with learnable inner lr (included in maml.py)
- [ ] 2.6 Create `maml_plus_plus.py` - MAML++ with gradual finetuning

## Phase 3: Prototypical Networks
- [x] 3.1 Create `protonet.py` - Basic Prototypical Networks
- [x] 3.2 Create `protonet.py` - Soft Prototypical Networks
- [x] 3.3 Create `protonet.py` - Variational ProtoNet
- [ ] 3.4 Create `squared_protonet.py` - Squared Euclidean ProtoNet

## Phase 4: Relation Networks
- [x] 4.1 Create `relationnet.py` - Basic Relation Network
- [x] 4.2 Create `relationnet.py` - Deep Relation Network
- [x] 4.3 Create `relationnet.py` - Multi-scale Relation Network
- [x] 4.4 Create `relationnet.py` - Attention-based Relation Network

## Phase 5: Matching Networks
- [x] 5.1 Create `matchingnet.py` - Basic Matching Networks
- [x] 5.2 Create `matchingnet.py` - Full-context Matching Networks
- [x] 5.3 Create `matchingnet.py` - Convolutional Matching Networks
- [x] 5.4 Create `matchingnet.py` - Imprinting weights for Matching Networks

## Phase 6: Episode Generation
- [x] 6.1 Create `episode_generator.py` - Base episode generator
- [x] 6.2 Create `episode_generator.py` - Task sampling utilities
- [x] 6.3 Create `episode_generator.py` - N-way K-shot sampler
- [x] 6.4 Create `episode_generator.py` - Class-incremental sampler
- [x] 6.5 Create `episode_generator.py` - Domain shift episode generator
- [ ] 6.6 Create `transformed_sampler.py` - Data augmentation in episodes

## Phase 7: Additional Algorithms
- [x] 7.1 Create `reptile.py` - Reptile algorithm
- [x] 7.2 Create `reptile.py` - Meta-learning baseline
- [ ] 7.3 Create `feat.py` - FEAT (Feature Transformer)
- [ ] 7.4 Create `cnaps.py` - CNAPS (Conditional Neural Adaptive Processes)
- [ ] 7.5 Create `leveraging_auxiliary.py` - Semi-supervised few-shot

## Phase 8: Training Utilities
- [x] 8.1 Create `episodic_trainer.py` - Episodic training loop
- [x] 8.2 Create `episodic_trainer.py` - Few-shot evaluation

## Phase 9: Base Encoders
- [x] 9.1 Create `encoders.py` - CNN/ResNet encoders for few-shot

## Phase 10: Testing & Integration
- [x] 10.1 Verify imports work correctly (syntax check)
