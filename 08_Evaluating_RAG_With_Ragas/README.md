
<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 8: Evaluating RAG with Ragas</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 8: Pre-Work](https://www.notion.so/Session-8-RAG-Evaluation-and-Assessment-1c8cd547af3d81d08f7cf5521d0253bb?pvs=4#1c8cd547af3d816583d6c23183b6f87f) | [Session 8: RAG Evaluation and Assessment](https://www.notion.so/Session-8-RAG-Evaluation-and-Assessment-1c8cd547af3d81d08f7cf5521d0253bb) | [Recording](https://us02web.zoom.us/rec/share/zlh_0cIuDgAtzeWsbQnScp-pjpdmVXvAAZv-Zi6sznb3IqCZTvcCQbfD-bXGUerr.fJP0Ywlyxlp0XPpe ) (*$EC4fQ1) | [Session 8 Slides](https://www.canva.com/design/DAGjadKGqcw/0Gff9K2EwbOb3lX14un3uw/edit?utm_content=DAGjadKGqcw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 8: RAG Evaluation and Assessment](https://forms.gle/ujAQLqx2ZHMWTUH79) | [AIE6 Feedback 4/24](https://forms.gle/wA7p89e6svCgjtr58) |

In today's assignment, we'll be creating Synthetic Data, and using it to benchmark (and improve) a LCEL RAG Chain.

- 🤝 Breakout Room #1
  1. Task 1: Installing Required Libraries
  2. Task 2: Set Environment Variables
  3. Task 3: Synthetic Dataset Generation for Evaluation using Ragas
  4. Task 4: Evaluating our Pipeline with Ragas
  5. Task 6: Making Adjustments and Re-Evaluating

  The notebook Colab link is located [here](https://colab.research.google.com/drive/1-t4POIFJI-SWF1lmoBOPETZZqgWCTV4Y?usp=sharing)

- 🤝 Breakout Room #2
  1. Task 1: Building a ReAct Agent with Metal Price Tool
  2. Task 2: Implementing the Agent Graph Structure
  3. Task 3: Converting Agent Messages to Ragas Format
  4. Task 4: Evaluating Agent Performance using Ragas Metrics
     - Tool Call Accuracy
     - Agent Goal Accuracy  
     - Topic Adherence

The notebook Colab link is located [here](https://colab.research.google.com/drive/1KQm7nA_zTaCyjaAeAacjqanMPv03um7T?usp=sharing)

## Ship 🚢

The completed notebook!

<details>
<summary>🚧 BONUS CHALLENGE 🚧 (OPTIONAL)</summary>

> NOTE: Completing this challenge will provide full marks on the assignment, regardless of the completion of the notebook. You do not need to complete this in the notebook for full marks.

##### **MINIMUM REQUIREMENTS**:

1. Baseline `LangGraph RAG` Application using `NAIVE RETRIEVAL`
2. Baseline Evaluation using `RAGAS METRICS`
  - [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)
  - [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html)
  - [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html)
  - [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html)
  - [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html)
3. Implement a `SEMANTIC CHUNKING STRATEGY`.
4. Create an `LangGraph RAG` Application using `SEMANTIC CHUNKING` with `NAIVE RETRIEVAL`.
5. Compare and contrast results.

##### **SEMANTIC CHUNKING REQUIREMENTS**:

Chunk semantically similar (based on designed threshold) sentences, and then paragraphs, greedily, up to a maximum chunk size. Minimum chunk size is a single sentence.

Have fun!
</details>

### Deliverables

- A short Loom of the notebook, and a 1min. walkthrough of the application in full

## Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped Synthetic Data Generation, benchmarking, and iteration with RAGAS & LangChain! 🎉🤖

🔍 Three Key Takeaways:
1️⃣ 
2️⃣ 
3️⃣ 

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```
