# PADE - Python Agent Development Framework (Updated)

## 📦 Versão Atualizada - Python 3.12+

Esta é a versão atualizada do PADE compatível com **Python 3.12+** e dependências modernas.

### ✨ Atualizações Principais

- **Python**: 3.12+ (anteriormente 3.7)
- **Flask**: 3.1.2 (anteriormente 1.1.1)
- **SQLAlchemy**: 2.0.44 (anteriormente 1.3.10)
- **Twisted**: 25.5.0 (anteriormente 19.7.0)
- **alchimia**: Removido (incompatível com SQLAlchemy 2.0)
  - Substituído por `twisted.enterprise.adbapi`

### 🚀 Instalação

#### Opção 1: Instalação via pip (desenvolvimento)
```bash
# Clone ou copie esta pasta
cd new_pade

# Crie um ambiente virtual
python3 -m venv padeenv
source padeenv/bin/activate  # Linux/Mac
# ou
padeenv\Scripts\activate  # Windows

# Instale em modo desenvolvimento
pip install -e .
```

#### Opção 2: Instalação direta
```bash
cd new_pade
pip install .
```

### 📋 Requisitos

- **Python**: >= 3.12
- **Sistema Operacional**: Linux, macOS, Windows
- **Dependências**: Instaladas automaticamente via `setup.py`

### 🎯 Uso Básico

#### 1. Criar agentes simples

```python
from pade.core.agent import Agent_
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import TimedBehaviour

class MyBehaviour(TimedBehaviour):
    def on_time(self):
        self.agent.send_message(...)

# Executar
if __name__ == '__main__':
    from pade.misc.utility import start_loop
    agents = [MyAgent()]
    start_loop(agents)
```

#### 2. Usar CLI do PADE

```bash
# Iniciar com interface web
pade start-runtime main.py --pade_ams --pade_web

# Sem AMS (comunicação peer-to-peer)
pade start-runtime main.py --no_pade_ams

# Porta customizada
pade start-runtime main.py --port 3000
```

#### 3. Acessar Interface Web

- **URL**: http://localhost:5001
- **Login**: pade_user / 12345 (configurável)
- **Recursos**:
  - Visualizar agentes registrados
  - Monitorar mensagens trocadas
  - Gerenciar sessões
  - Ver estatísticas

### 🔧 Configuração (config.json)

```json
{
  "session": {
    "username": "pade_user",
    "email": "pade_user@pade.com",
    "password": "12345"
  },
  "pade_ams": {
    "launch": true,
    "host": "localhost",
    "port": 8000
  },
  "pade_web": {
    "active": true,
    "host": "0.0.0.0",
    "port": 5001
  },
  "pade_sniffer": {
    "active": true,
    "port": 8001
  }
}
```

### 🆕 Novos Recursos

#### Comunicação sem AMS
A versão atualizada permite comunicação **peer-to-peer direta** sem necessidade do AMS:

```python
# Os agentes se comunicam diretamente se tiverem host e port definidos
sensor_aid = AID('sensor@localhost:2001')
central.send(message, receiver=sensor_aid)
```

#### Compatibilidade macOS
- Porta web padrão alterada de **5000** para **5001** (conflito com AirPlay)
- Subprocess corrigido para suportar espaços em caminhos

### 📚 Documentação Completa

Consulte os seguintes documentos na pasta raiz:

1. **ATUALIZACAO_CONCLUIDA.md** - Status da migração
2. **RESUMO_MIGRACAO.md** - Resumo executivo
3. **MODIFICACOES_PADE.md** - Detalhes técnicos das mudanças
4. **GUIA_PADE_WEB.md** - Como usar a interface web
5. **CHECKLIST_VALIDACAO_FINAL.md** - Testes de validação
6. **PROBLEMAS_COMUNS.md** - Troubleshooting

### 🐛 Problemas Conhecidos

#### Porta 5000 em uso (macOS)
O AirPlay usa a porta 5000 por padrão no macOS. Use:
```bash
pade start-runtime main.py --web_port 5001
```

#### Espaços em caminhos
Corrigido! A versão atualizada suporta caminhos com espaços.

#### SQLAlchemy 2.0
Todos os métodos foram atualizados para usar context managers:
```python
with engine.connect() as conn:
    result = conn.execute(query)
```

### 🧪 Testes

```bash
# Testar instalação
python -c "import pade; print(pade.__version__)"

# Executar exemplos
cd examples
python agent_example_1.py
```

### 📄 Licença

MIT License - Copyright (c) 2019 Lucas S Melo

### 👥 Autores

**Versão Original:**
- Electric Smart Grid Group - GREI
- Federal University of Ceara - UFC - Brazil
- Laboratory of Applied Artificial Intelligence - LAAI
- Federal University of Para - UFPA

**Atualização Python 3.12+:**
- Migração realizada em Novembro/2025
- Compatibilidade com dependências modernas

### 🔗 Links

- **GitHub Original**: https://github.com/grei-ufc/pade
- **Documentação**: Consulte arquivos `.md` na raiz do projeto

---

## 📦 Estrutura de Arquivos

```
new_pade/
├── pade/                 # Código fonte do framework
│   ├── acl/             # ACL messages e AID
│   ├── behaviours/      # Comportamentos dos agentes
│   ├── core/            # Core (Agent, AMS, Sniffer)
│   ├── web/             # Interface web Flask
│   └── misc/            # Utilitários
├── setup.py             # Script de instalação
├── requirements.txt     # Dependências
├── LICENSE              # Licença MIT
└── README.md            # Este arquivo
```

---

## 🚀 Início Rápido

```bash
# 1. Instalar
pip install .

# 2. Criar agente (main.py)
from pade.misc.utility import start_loop
from pade.core.agent import Agent_

agents = [Agent_('agent1@localhost:2000')]
start_loop(agents)

# 3. Executar
pade start-runtime main.py --pade_ams
```

Acesse http://localhost:5001 e veja seus agentes em ação! 🎉
