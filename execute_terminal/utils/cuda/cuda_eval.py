
import torch



def verify_mem():
    print("="*80)
    print("📊 DIAGNÓSTICO DO SISTEMA GPU")
    print("="*80)
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print(f"GPU encontrada: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma'}")

    if torch.cuda.is_available():
        # Limpar cache antes de começar
        print("\n🧹 Limpando cache de GPU...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Informações de memória
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1e9
        print(f"Memória total da GPU: {total_memory:.2f} GB")
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Memória alocada: {allocated:.2f} GB")
        print(f"Memória reservada: {reserved:.2f} GB")
        print(f"Memória livre: {(total_memory - reserved):.2f} GB")
        
        # Verificar compatibilidade
        print(f"\nVersão CUDA: {torch.version.cuda}")
        print(f"Versão cuDNN: {torch.backends.cudnn.version()}")
        print(f"cuDNN habilitado: {torch.backends.cudnn.enabled}")
        
    else:
        print("⚠️  CUDA não disponível! Usando CPU (muito lento).")

    print("="*80 + "\n")