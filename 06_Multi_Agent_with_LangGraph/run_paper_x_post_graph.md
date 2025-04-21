```mermaid
%%{
  init: {
    'theme': 'dark'
  }
}%%
graph TD
    Start["Script Execution Start<br>(if __name__ == '__main__')"] -->|Sets up initial state| InvokeGraph{Invoke final_graph};

    InvokeGraph --> MetaSupervisor["Meta-Supervisor Node<br>(meta_supervisor_agent)"];

    MetaSupervisor -- Route to 'Research' --> ResearchNode["Research Team Node<br>(runs run_research)"];
    ResearchNode -- Invokes --> ResearchGraph[("Research Graph<br>(fetch_summary using arxiv_tool)")];
    ResearchGraph -- Returns summary --> ResearchNode;
    ResearchNode -- Returns to --> MetaSupervisor;

    MetaSupervisor -- Route to 'Authoring' --> AuthoringNode["Authoring Team Node<br>(runs run_authoring)"];
    AuthoringNode -- Invokes --> AuthoringGraph[("Authoring Graph<br>(X_Post_Drafter, Conciseness_Editor,<br>authoring_supervisor using file_tools)")];
    AuthoringGraph -- Writes x_draft.txt --> AuthoringNode;
    AuthoringNode -- Returns to --> MetaSupervisor;

    MetaSupervisor -- Route to 'Verification' --> VerificationNode["Verification Team Node<br>(runs run_verification)"];
    VerificationNode -- Invokes --> VerificationGraph[("Verification Graph<br>(FactChecker, X_StyleChecker,<br>verification_supervisor, aggregate_results using file_tools)")];
    VerificationGraph -- Returns verification status --> VerificationNode;
    VerificationNode -- Returns to --> MetaSupervisor;

    MetaSupervisor -- Route to 'FINISH' --> EndGraph([END of final_graph]);
    EndGraph --> PrintResults[Script Prints Results & Exits];

    %% Styling removed as dark theme provides defaults
