```mermaid
%%{ init: { 'theme': 'dark' } }%%
graph TD
    %% Node Definitions First
    Start([Start Script])
    Research["Research<br/><i>Fetch summary for '2404.19553'</i>"]
    Supervisor{"Supervisor<br/><i>Decides Next Step</i>"}
    Authoring["Authoring<br/><i>Create initial draft</i>"]
    Verification["Verification<br/><i>Checking draft...</i>"]
    Authoring_Rev1["Authoring<br/><i>Revise draft (Attempt 1)</i>"]
    Verification_Rev1["Verification<br/><i>Checking draft...</i>"]
    End([End Script])

    %% Edges
    Start -->|Step 1: Fetch Summary| Research;
    Research -->|Step 2| Supervisor;
    Supervisor -->|Step 3: Route to Authoring| Authoring;
    Authoring -->|Step 4| Supervisor;
    Supervisor -->|Step 5: Route to Verification| Verification;
    Verification -->|Step 6: Result: Verified=False| Supervisor;
    Supervisor -->|Step 7: Route to Authoring Rev 1| Authoring_Rev1;
    Authoring_Rev1 -->|Step 8| Supervisor;
    Supervisor -->|Step 9: Route to Verification| Verification_Rev1;
    Verification_Rev1 -->|Step 10: Result: Verified=True| Supervisor;
    Supervisor -->|Step 11: Route to FINISH| End;

    %% Styling (Optional - Dark theme handles defaults)
    classDef default fill:#2b2b2b,stroke:#ccc,color:#ccc;
    classDef teamNode fill:#3a3a50,stroke:#88f;
    classDef supervisorNode fill:#503a3a,stroke:#f88;
    classDef startEndNode fill:#3a503a,stroke:#8f8;

    class Start,End startEndNode;
    class Research,Authoring,Authoring_Rev1,Verification,Verification_Rev1 teamNode;
    class Supervisor supervisorNode;