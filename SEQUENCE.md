```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant ExternalAPI

    User->>Frontend: Initiates Action (e.g., Request Data)
    Frontend->>Backend: Sends API Request
    %% Database interactions and login removed
    Backend->>ExternalAPI: (Optional) Calls External Service
    ExternalAPI-->>Backend: Returns Data/Status
    %% Data handled in-memory in Backend
    Backend-->>Frontend: Returns Response (Data/Status)
    Frontend-->>User: Displays Results/Updates UI
```
