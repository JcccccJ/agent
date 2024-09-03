# 配置环境

1. 在conda中新建虚拟环境，python版本选择3.9

```
conda create --name your_env_name python=3.9
```

2. 安装所需要的包

   - 从源码安装agentscope

   ```cmd
   git clone https://github.com/modelscope/agentscope.git
   cd agentscope
   # 针对本地化的multi-agent应用
   pip install -e .
   # 为分布式multi-agent应用
   pip install -e .[distribute]  # 在Mac上使用pip install -e .\[distribute\]
   ```

   - 安装llama等包

   ```cmd
   pip install llama-index==0.10.30 llama-index-readers-docstring-walker==0.1.3 tree-sitter==0.21.3 tree-sitter-languages==1.10.2`
   ```

# 配置config文件

- model_config.json

```json
[
  {
    "model_type": "dashscope_text_embedding",
    "config_name": "qwen_emb_config",
    "model_name": "text-embedding-v2",
    "api_key": "{you_api_key}"
  },
  {
    "model_type": "dashscope_chat",
    "config_name": "qwen_config",
    "model_name": "qwen-max",
    "api_key": "{you_api_key}"
  }
]
```



- knowledge_config.json

```json
[
{
  "knowledge_id": "agent_llm_rag",
  "emb_model_config_name": "qwen_emb_config",
  "data_processing": [
    {
      "load_data": {
        "loader": {
          "create_object": true,
          "module": "llama_index.core",
          "class": "SimpleDirectoryReader",
          "init_args": {
            "input_dir": "./data/pdf",
            "required_exts": [".pdf"]
          }
        }
      }
    }
  ]
}
]
```

**更多介绍请看[agentscope官方文档](http://doc.agentscope.io/zh_CN/tutorial/210-rag.html)**

- agent_config.json

```json
[
  {
    "class": "LlamaIndexAgent",
    "args": {
      "name": "knowledge-Assistant",
      "description": "Pdf-Assistant is an agent that can provide answer based on Chinese Large language model material, mainly the pdf files. It can answer general questions about AgentScope.",
      "sys_prompt": "You're an assistant helping new users to use AgentScope. The language style is helpful and cheerful. You generate answers based on the provided context. The answer is expected to be no longer than 100 words. If the key words of the question can be found in the provided context, the answer should contain the section name which contains the answer. For example, 'You may refer to SECTION_NAME for more details.'",
      "model_config_name": "qwen_config",
      "knowledge_id_list": ["agent_llm_rag"],
      "similarity_top_k": 5,
      "log_retrieval": false,
      "recent_n_mem_for_retrieve": 1
    }
  },
  {
    "class": "DialogAgent",
    "args": {
      "name": "Agent-Guiding-Assistant",
      "sys_prompt": "You're an assistant guiding the user to specific agent for help. The answer is in a cheerful styled language. The output starts with appreciation for the question. Next, rephrase the question in a simple declarative Sentence for example, 'I think you are asking...'. Last, if the question is about agent or large language model, output '@ knowledge-Assistant, I think you are more suitable for the question, can you tell us more about it'. The answer is expected to be only one sentence",
      "model_config_name": "qwen_config",
      "use_memory": false
    }
  }
]
```



# 编写主要代码

- groupchat

```python
# -*- coding: utf-8 -*-
import re
from typing import Sequence


def select_next_one(agents: Sequence, rnd: int) -> Sequence:
    """
    选择下一个agent
    """
    return agents[rnd % len(agents)]


def filter_agents(string: str, agents: Sequence) -> Sequence:
    """
    这个函数过滤输入字符串中出现的给定名称以'@'作为前缀，并返回找到的名称的列表。
    """
    if len(agents) == 0:
        return []

    # 创建一个匹配@后跟任何候选名称的模式
    pattern = (
            r"@(" + "|".join(re.escape(agent.name) for agent in agents) + r")\b"
    )

    # 找出字符串中出现的所有模式
    matches = re.findall(pattern, string)

    # 创建一个将代理名称映射到代理对象的字典，以便快速查找
    agent_dict = {agent.name: agent for agent in agents}

    # 返回保持顺序的匹配代理对象的列表
    ordered_agents = [
        agent_dict[name] for name in matches if name in agent_dict
    ]
    return ordered_agents
```

- mian

```python
# -*- coding: utf-8 -*-
import json

import agentscope
from agentscope.rag import KnowledgeBank
from agentscope.agents import UserAgent

from groupchat import filter_agents

# 接待用户的agent选择合适的Rag_agent回答的提示词
AGENT_CHOICE_PROMPT = """
There are following available agents. You need to choose the most appropriate
agent(s) to answer the user's question.

agent descriptions:{}

First, rephrase the user's question, which must contain the key information.
The you need to think step by step. If you believe some of the agents are
good candidates to answer the question (e.g., AGENT_1 and AGENT_2), then
you need to follow the following format to generate your output:

'
Because $YOUR_REASONING.
I believe @AGENT_1 and @AGENT_2 are the most appropriate agents to answer
your question.
'
"""


def main():
    # 加载模型配置
    with open("configs/model_config.json", "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    # 加载agent配置
    with open("configs/agent_config.json", "r", encoding="utf-8") as f:
        agent_configs = json.load(f)

    agent_list = agentscope.init(
        model_configs=model_configs,
        agent_configs=agent_configs,
        project="Conversation with RAG agents",
    )
    rag_agent_list = agent_list[:1]
    guide_agent = agent_list[1]

    # 通过加载配置文件，可以对知识库进行配置
    knowledge_bank = KnowledgeBank(configs="./configs/knowledge_config.json")

    # 或者，输入配置来给RAG添加数据
    # knowledge_bank.add_data_as_knowledge(
    #     knowledge_id="agent_pdf_rag",
    #     emb_model_name="qwen_emb_config",
    #     data_dirs_and_types={
    #         "./data/pdf": [".pdf"],
    #     },
    # )

    # 通过知识库knowledgebank, 给对应的rag_agent配备响应的知识
    for agent in rag_agent_list:
        knowledge_bank.equip(agent, agent.knowledge_id_list)

    # 或者在初始化agent的时候, 配备知识knowledge 
    #
    # ```
    # knowledge = knowledge_bank.get_knowledge(knowledge_id)
    # agent = LlamaIndexAgent(
    #   name="rag_worker",
    #   sys_prompt="{your_prompt}",
    #   model_config_name="{your_model}",
    #   knowledge_list=[knowledge], # provide knowledge object directly
    #   similarity_top_k=3,
    #   log_retrieval=False,
    #   recent_n_mem_for_retrieve=1,
    # )
    # ```

    rag_agent_names = [agent.name for agent in rag_agent_list]
	# 生成对所有rag_agent的描述
    rag_agent_descriptions = [
        "agent name: "
        + agent.name
        + "\n agent description："
        + agent.description
        + "\n"
        for agent in rag_agent_list
    ]
	
    # 生成guide_agent的prompt
    guide_agent.sys_prompt = (
            guide_agent.sys_prompt
            + AGENT_CHOICE_PROMPT.format(
        "".join(rag_agent_descriptions),
    )
    )

    user_agent = UserAgent()
    while True:
        # 用户输入问题 --> 接待agent分析问题并选择合适的agent --> 选择的agent进行回答 
        x = user_agent()
        x.role = "user"  # to enforce dashscope requirement on roles
        if len(x.content) == 0 or str(x.content).startswith("exit"):
            break
        speak_list = filter_agents(x.content, rag_agent_list)
        if len(speak_list) == 0:
            guide_response = guide_agent(x)
            speak_list = filter_agents(
                guide_response.content,
                rag_agent_list,
            )
        agent_name_list = [agent.name for agent in speak_list]
        for agent_name, agent in zip(agent_name_list, speak_list):
            if agent_name in rag_agent_names:
                agent(x)


if __name__ == "__main__":
    main()
```



