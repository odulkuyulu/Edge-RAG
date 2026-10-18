# Partner Enablement Lab: Edge RAG with Local Azure AI Containers

## 1. Session purpose and boundaries

**Audience:** solution architects, technical partners, and engineers familiar with basic Azure and Windows administration.

**Outcome:** participants run an English-language document RAG demo on one x64 Windows host using Docker Desktop, local Azure AI containers, Qdrant, Ollama, and a Streamlit application.

**Lab duration:** approximately 90-120 minutes **after prework**. Downloading images and models can take considerably longer; complete downloads before the session.

### Explain these distinctions before starting

| Term | What this lab actually provides |
|---|---|
| Local inference | Document extraction, language enrichment, embeddings, retrieval, and generation run on the chosen host. |
| Azure-hosted VM | If the host is an Azure VM, processing is local to that VM, not on the participant's laptop or on-premises. |
| Connected Azure AI containers | The Azure AI containers still require valid Azure resources, keys, and billing connectivity. Usage can incur charges. |
| Fully disconnected operation | **Not covered.** It requires a separately approved licensing/configuration workflow; pulling these images does not grant disconnected use. |
| Language coverage | **English only for this container stack.** Do not promise the cloud version's Arabic enrichment coverage. |
| Production readiness | This is an enablement demo, not a hardened or highly available production deployment. |

Application inference requests are routed to local endpoints. Review container terms, diagnostics, and network requirements separately; do not claim that the host has no outbound traffic.

### Success criteria

- Eight Docker containers are running.
- Every local inference check passes, including PDF analysis, embeddings, and generation.
- The unchanged sample dataset produces eight English chunks from two documents.
- The Arabic sample is explicitly skipped during English-only indexing.
- A Streamlit query retrieves sources and generates an answer.
- The application is accessed from the Docker host's browser without exposing service ports publicly.

## 2. Facilitator preparation: complete before inviting partners

### 2.1 Prepare a reviewed workshop release

This guide depends on changes that may exist only in a development worktree. **Do not assume that cloning the repository's default branch includes them.**

Publish a reviewed workshop branch/tag/commit or distribute an approved source ZIP containing:

| Required file or folder | Purpose |
|---|---|
| [compose.local-ai.yaml](./compose.local-ai.yaml) | Eight-service Docker stack |
| [Start-LocalApp.ps1](./Start-LocalApp.ps1) | Selects local endpoints and English-only indexing |
| [src/container_language.py](./src/container_language.py) | Routes each Language feature to its container |
| [src/indexer.py](./src/indexer.py), [src/retriever.py](./src/retriever.py) | Local-container integration |
| [src/enhanced_indexer.py](./src/enhanced_indexer.py), [src/enhanced_app.py](./src/enhanced_app.py), [src/enhanced_retriever.py](./src/enhanced_retriever.py) | Indexing and UI workflow |
| [tests/verify_local_services.py](./tests/verify_local_services.py) | Real inference and Streamlit validation |
| [tests/test_container_language.py](./tests/test_container_language.py) | Routing and regression tests |
| [data/](./data) | Unchanged workshop samples |

Exclude `.env`, API keys, virtual environments, caches, and private participant documents from distributed packages.

Give participants:

- The workshop release URL and exact commit/tag, or the approved ZIP.
- Repository access if required.
- Their host access instructions.
- A secure method to obtain their own Azure resource endpoints and keys.
- An approved package-feed/proxy configuration.
- A designated lab owner for Azure permissions and governance issues.

Record the tested release, dependency versions, and container image digests in your session preparation notes. Several images use mutable `latest` tags; do not update the stack in the middle of a workshop.

### 2.2 Prepare hosts

Use one dedicated host per participant or small team. The supplied Compose project name and ports are fixed; it is not a multi-team stack for a shared host.

| Requirement | Workshop recommendation |
|---|---|
| Architecture | x64/AMD64; not an ARM64 laptop |
| Windows host | Windows with supported Docker Desktop/WSL2 configuration |
| CPU | 16 vCPUs recommended for this lab |
| Host memory | 96-128 GiB recommended |
| Docker memory | Approximately 64 GiB or more; configured container limits total 58 GiB, excluding Qdrant and runtime overhead |
| Storage | At least a 256 GiB OS disk or a suitably sized persistent data disk; plan at least 150 GiB free for a cold setup and more if retaining multiple image stores |
| Python | Python 3.12 x64 recommended; the reference VM also ran successfully with Python 3.11 x64 |
| Tools | Git, Docker Desktop with Compose, PowerShell, and a working way to download Ollama models |
| GPU | Not required for this CPU-based demo; response time depends on the host |

For Azure VMs, have the infrastructure owner confirm nested-virtualization support and the Docker Desktop/WSL2 prerequisites for the selected VM size.

Use persistent storage for images and models. Do not put important Docker data on an Azure temporary disk, commonly labeled `Temporary Storage`. Drive letters vary between hosts.

If using Bastion:

1. Connect to the VM's Windows desktop.
2. Sign in as the intended lab user.
3. Start Docker Desktop in that interactive session.
4. Run **all lab commands inside the VM**, unless a step explicitly says otherwise.
5. Open the final UI in a browser **inside that VM**.

`localhost` always means the machine running the command or browser. This lab does not require SSH, a public VM IP, or public container ports.

### 2.3 Prepare Azure billing resources and governance

Provision or assign access to:

- An **Azure AI Language** resource.
- An **Azure AI Document Intelligence** resource.
- A region and pricing tier eligible for the chosen container images, according to current Microsoft documentation.

The person configuring the lab must be able to obtain the required endpoints and keys through approved access.

**Governance gate:** this Compose stack uses API keys for Azure container billing. If your organization disables local/key authentication:

1. Stop at this gate.
2. Ask the resource/security owner to confirm the supported container authentication path.
3. Obtain an approved, appropriately scoped exception if one is permitted.
4. Otherwise redesign the authentication approach before continuing.

Do not copy another tenant's exclusion tags, disable governance policies, or assume an app-level Entra sign-in automatically configures container billing. Tenant-specific exception mechanisms are not general workshop prerequisites.

### 2.4 Complete network and licensing checks

Arrange approved outbound access for:

- Microsoft Container Registry and its required download endpoints.
- Docker Hub and the required registry/CDN endpoints.
- The approved Python package feed.
- Ollama's model registry and model download endpoints.
- Azure AI billing endpoints.
- Initial NLTK tokenizer/corpus downloads.

Use your organization's current allowlists and proxy/certificate guidance rather than assuming one domain covers every download.

Starting the Compose stack sets `Eula=accept` for the Azure AI containers. Confirm that the organization is authorized to accept the applicable terms, and review Ollama/model licenses and Docker Desktop licensing as appropriate.

**Facilitator checkpoint:** rehearse every step on a fresh host or a fresh lab volume before the session. Use only approved sample data.

## 3. Suggested delivery agenda

| Segment | Approximate time | Presenter focus |
|---|---:|---|
| Architecture and scope | 10 min | Local inference versus disconnected licensing |
| Host and source preflight | 10 min | Correct machine, architecture, storage, release |
| Python and configuration | 15 min | Environment isolation and secret handling |
| Models and container startup | 15 min | Model storage, ports, service dependencies |
| Real inference validation | 10 min | Prove services work, not merely that processes exist |
| Document indexing | 10 min | Extraction, chunking, enrichment, vector persistence |
| Streamlit demonstration | 15 min | Grounded answers and source inspection |
| Operations and Q&A | 15 min | Restart, troubleshooting, costs, limitations |

Cold downloads are prework, not a timed part of this agenda.

## 4. Architecture to explain

```text
Windows host / VM
|
+-- Streamlit UI and Python application :8503
|   |
|   +-- Document Intelligence :5010
|   |   +-- Layout container (internal Compose network)
|   |
|   +-- Language detection :5000
|   +-- Named entity recognition :5001
|   +-- Key phrases :5002
|   +-- English sentiment :5003
|   |
|   +-- Ollama :11435 -> bge-m3 and qwen2.5:0.5b
|   +-- Qdrant :6333/:6334 -> persistent vector data
|
+-- Azure AI containers -> Azure billing connectivity
```

| Compose service | Host port | Container port | Role |
|---|---:|---:|---|
| `language` | 5000 | 5000 | Language detection |
| `ner` | 5001 | 5000 | Named entity recognition |
| `keyphrase` | 5002 | 5000 | Key-phrase extraction |
| `sentiment` | 5003 | 5000 | English sentiment |
| `document` | 5010 | 5050 | General Document model |
| `layout` | Not published | 5000 | Document model dependency |
| `qdrant` | 6333, 6334 | 6333, 6334 | REST/gRPC vector database |
| `ollama` | 11435 | 11434 | Embeddings and generation |

All published Compose ports bind to `127.0.0.1`. The Python UI also binds to loopback.

**Talk track:** a Language container provides a feature, not necessarily the entire cloud Language API. The application must route detection, entities, key phrases, and sentiment separately.

## 5. Step 1: verify the host

Open PowerShell on the chosen Docker host.

```powershell
Get-CimInstance Win32_ComputerSystem |
    Select-Object SystemType, TotalPhysicalMemory

Get-Volume |
    Select-Object DriveLetter, FileSystemLabel, Size, SizeRemaining

docker version
docker info
docker compose version
```

**Expected checkpoint**

- System is x64.
- `docker version` contains both **Client** and **Server** sections.
- Docker reports `OSType: linux` and `Architecture: x86_64`/AMD64.
- Docker has the required memory allocation.
- The disk containing Docker data has sufficient free space.

A Docker client version alone is **not** evidence of a working engine.

If the engine is not ready, start Docker Desktop and wait for **Engine running**. Do not continue into image downloads on an almost-full disk.

## 6. Step 2: obtain the workshop source

For a Git-based workshop, use a new directory:

```powershell
$LabRoot = Join-Path $env:USERPROFILE 'edge-rag-workshop'
git clone "https://github.com/odulkuyulu/Edge-RAG.git" $LabRoot
Set-Location $LabRoot
```

If that directory already exists, inspect it first or choose another directory. Do not delete an existing project to make the command succeed.

Select the facilitator's published workshop release:

```powershell
$WorkshopRelease = Read-Host 'Enter the facilitator-provided workshop tag or commit'
git checkout --detach $WorkshopRelease
git rev-parse HEAD
```

Alternatively, extract the approved workshop ZIP into a new directory and open PowerShell there. Git-only commands do not apply to an extracted ZIP.

Verify that the required workshop files exist:

```powershell
Test-Path .\compose.local-ai.yaml
Test-Path .\Start-LocalApp.ps1
Test-Path .\src\container_language.py
Test-Path .\tests\verify_local_services.py
Test-Path .\data\ai_en.txt
Test-Path .\data\eventss.pdf
```

**Expected checkpoint:** all values are `True`. If not, obtain the correct workshop package before proceeding.

## 7. Step 3: create the Python environment

Verify the selected interpreter is x64:

```powershell
py -3.12 -c "import platform; print(platform.python_version()); print(platform.machine())"
```

If Python 3.12 x64 is unavailable, install it through the organization's approved software process. Use an approved x64 Python 3.11 environment only if the facilitator has validated that combination.

Create the environment:

```powershell
py -3.12 -m venv venv
```

Use the environment's interpreter explicitly; activation is not required:

```powershell
.\venv\Scripts\python.exe -m pip install ollama "qdrant-client<1.16" streamlit rank-bm25 nltk azure-ai-formrecognizer "azure-core==1.30.1" "azure-ai-textanalytics==5.3.0" PyPDF2 python-dotenv requests numpy pytest

.\venv\Scripts\python.exe -m pip check
```

**Why this dependency list?**

- It covers the enhanced demo's runtime and validation.
- The broader repository requirements also contain packages not needed for this lab.
- The retrievers use Qdrant's legacy `search()` API, removed from client 1.16; keep the client below 1.16.
- This is not a complete dependency lock. Use the facilitator's tested release/environment for reproducibility.

If package installation fails, capture the error and use the approved feed/proxy remediation. Do not disable TLS verification or bypass organizational network controls.

**Expected checkpoint:** installation succeeds and `pip check` reports no broken requirements.

## 8. Step 4: prepare the Ollama models

The container reads an existing host model directory **read-only**. It cannot download models into that mount.

### Option A: download with a working host Ollama installation

Install the official approved Ollama distribution and start its desktop application. If no server is running, `ollama serve` can run in a separate terminal.

In the setup terminal:

```powershell
$env:OLLAMA_HOST = 'http://127.0.0.1:11434'
ollama pull bge-m3
ollama pull qwen2.5:0.5b
ollama list
```

The host server on 11434 is used for preparation. The workshop application's isolated container will use 11435.

### Option B: reuse approved, already-downloaded models

If the model files already exist, do not download them again. Point Compose to the actual populated model directory, preserving both manifests and blobs.

```powershell
$env:OLLAMA_MODELS_PATH = Join-Path $env:USERPROFILE '.ollama\models'
Test-Path -LiteralPath $env:OLLAMA_MODELS_PATH
```

If your models are stored elsewhere, set `OLLAMA_MODELS_PATH` to that actual directory.

**Expected checkpoint:** the directory exists and contains the two required models. Later validation must prove real embedding and generation requests; a model list alone is insufficient.

Environment variables set in one PowerShell window are not automatically shared with another. Set this variable again in each Compose terminal, or save its actual value in the private `.env` as described next.

## 9. Step 5: create the private environment file

From the repository root:

```powershell
notepad .env
```

Save as exactly `.env`, not `.env.txt`. Use **All files** in the Save dialog if necessary.

Use this template with the participant's own approved values:

```dotenv
# Cloud endpoints are retained for container billing.
AZURE_LANGUAGE_ENDPOINT=https://REPLACE_WITH_LANGUAGE_RESOURCE.cognitiveservices.azure.com/
AZURE_LANGUAGE_BILLING_URL=https://REPLACE_WITH_LANGUAGE_RESOURCE.cognitiveservices.azure.com/
AZURE_LANGUAGE_KEY=REPLACE_WITH_LANGUAGE_KEY

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://REPLACE_WITH_DOCUMENT_RESOURCE.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_BILLING_URL=https://REPLACE_WITH_DOCUMENT_RESOURCE.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=REPLACE_WITH_DOCUMENT_KEY

# Use the actual populated model directory on this Windows host.
OLLAMA_MODELS_PATH=C:\Users\REPLACE_WITH_WINDOWS_USER\.ollama\models

QDRANT_URL=http://127.0.0.1:6333
EMBEDDING_MODEL=bge-m3
EMBEDDING_SIZE=1024
CHUNK_SIZE=200
CHUNK_OVERLAP=50
```

Replace every `REPLACE_WITH_...` value before continuing. URLs must be plain URLs, not Markdown links such as `[https://...](https://...)`.

**Important:** do not change the billing URLs to localhost. [Start-LocalApp.ps1](./Start-LocalApp.ps1) overrides application inference endpoints in its child process while Compose retains the cloud billing configuration.

On a Git checkout, verify exclusion without printing file contents:

```powershell
git check-ignore .env
git ls-files -- .env
```

**Expected checkpoint:** the first command prints `.env`; the second prints nothing.

Never display keys on the shared screen, put them in slides/chat, or include `.env` in a source package. Restrict access to the file. If a key is exposed, have the resource owner rotate it.

Do not run bare `docker compose config` or share full `docker inspect` output: expanded container environments can expose billing keys. Use `config --quiet` and narrow diagnostic queries.

## 10. Step 6: start the container stack

First check for conflicting lab ports:

```powershell
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in 5000,5001,5002,5003,5010,6333,6334,8503,11435 } |
    Select-Object LocalAddress, LocalPort, OwningProcess
```

If a port is occupied, identify its owner. Do not stop another participant's service or kill processes by name. Resolve conflicts with the facilitator before continuing.

Validate the configuration without printing expanded secrets:

```powershell
docker compose -f .\compose.local-ai.yaml config --quiet
```

Start the stack:

```powershell
docker compose -f .\compose.local-ai.yaml up -d
docker compose -f .\compose.local-ai.yaml ps
```

Large image downloads and first model loads take time. Compose reporting a container as `Up` does **not** prove its inference API is ready.

**Expected checkpoint:** eight services are running: `language`, `ner`, `keyphrase`, `sentiment`, `document`, `layout`, `qdrant`, and `ollama`.

**Presenter notes**

- General Document depends on Layout; both are necessary.
- Document Intelligence publishes host 5010 to container 5050.
- Feature containers listen on container port 5000, even when the host port differs.
- The Ollama container uses port 11435 so it does not replace a host installation on 11434.
- Qdrant data is kept in a project-specific named volume.

## 11. Step 7: validate real local inference

Run from the repository root:

```powershell
.\venv\Scripts\python.exe .\tests\verify_local_services.py
```

The script explicitly selects local endpoints. It checks:

1. Language detection.
2. Named entity recognition.
3. Key-phrase extraction.
4. Sentiment analysis.
5. Actual extraction from the sample PDF through Document Intelligence.
6. Qdrant health.
7. Ollama model availability.
8. A real 1,024-dimensional embedding.
9. A real generated response.

**Expected checkpoint:** every check prints `PASS` and the process exits successfully.

If this fails, stop and troubleshoot the failing service before indexing. Do not count a running container, model listing, successful import, or health endpoint as a substitute for actual inference.

Optional regression tests:

```powershell
.\venv\Scripts\python.exe -m pytest .\tests\test_container_language.py -q
```

The current workshop routing suite contains 11 passing tests, including checks that cloud behavior is preserved when local mode is not enabled and local PDF failures do not silently switch extractors.

## 12. Step 8: index the English sample documents once

Use the local launcher, not the generic cloud-mode command:

```powershell
.\Start-LocalApp.ps1 -PythonPath .\venv\Scripts\python.exe -IndexOnly
```

For the unchanged workshop samples and validated container versions, expect:

| Document | Expected outcome |
|---|---|
| `ai_en.txt` | Five English chunks indexed |
| `eventss.pdf` | Three English chunks indexed |
| Arabic PDF sample | Explicitly skipped because English-only indexing is selected |

The resulting English collection is `rag_docs_en_bge-m3`.

**Expected checkpoint:** eight successfully stored English chunks, no failed English chunks, and an explicit skip for the non-English sample.

Explain the processing flow:

```text
Document -> local extraction -> language detection -> intelligent chunking
         -> local language enrichment -> local embedding -> Qdrant
```

**Do not rerun indexing casually.** The current indexer inserts new point IDs, so repeating the command adds duplicates rather than replacing existing chunks.

## 13. Step 9: prove the complete RAG flow

Before starting the persistent UI:

```powershell
.\venv\Scripts\python.exe .\tests\verify_local_services.py --rag
```

In addition to the service checks, this verifies:

- Exactly eight points in the unchanged English sample collection.
- The two expected source filenames.
- Rendering of the local-mode Streamlit notice.
- A real query submitted through Streamlit's testing interface.
- Nonzero sources found and sources used for generation.

**Expected checkpoint**

```text
Local indexed English documents: PASS (8 chunks)
Local Streamlit retrieval and generation flow: PASS
```

The exact-eight assertion is a workshop fixture check, not a universal requirement for arbitrary document sets. Different fixtures, repeated indexing, or changed extraction/model/dependency versions require investigation and an intentionally revised baseline. Do not weaken an assertion simply to hide an unexplained failure.

## 14. Step 10: launch the UI

Open a second PowerShell window on the same host and navigate to the same repository directory.

```powershell
.\Start-LocalApp.ps1 -PythonPath .\venv\Scripts\python.exe
```

Keep this terminal open.

In the host's browser, open:

```text
http://localhost:8503
```

For a VM lab, this means the browser **inside the Bastion desktop**, not the participant's laptop browser.

From another PowerShell window, verify health:

```powershell
Invoke-RestMethod http://127.0.0.1:8503/_stcore/health
```

**Expected checkpoint:** `ok`, a visible local-Docker/English-only notice, and the enhanced RAG interface.

Do not start Streamlit with an unrelated command that omits the launcher's environment settings; the original `.env` deliberately retains cloud billing endpoints.

### Guided demonstration

1. Keep the embedding model set to **bge-m3**.
2. Keep the generation model set to **qwen2.5:0.5b**.
3. Leave document filtering at **All** and minimum importance at **0.0**.
4. Ask: **What is the Microsoft G42 partnership about?**
5. Run **Enhanced Search & Generate**.
6. Expand the source sections.
7. Compare the answer with the retrieved source text.
8. Ask: **How much did Microsoft invest in G42?**
9. Discuss whether the sources actually support the generated claims.

Suggested talk track:

> The application retrieves relevant local chunks and provides them as context to a local generation model. Source inspection is essential: retrieval and generation do not guarantee correctness. The UI's confidence indicator is a heuristic, not a calibrated probability.

Keep the initial lab focused on bundled English documents. Validate any additional uploads, models, document formats, or languages separately before adding them to the partner session.

## 15. Daily restart and session operation

After restarting Windows:

1. Sign into the lab user's Windows desktop.
2. Start Docker Desktop and wait for the engine.
3. Start or reconcile the stack:

   ```powershell
   docker compose -f .\compose.local-ai.yaml up -d
   docker compose -f .\compose.local-ai.yaml ps
   ```

4. Run the service verifier if the environment changed.
5. Start the UI with [Start-LocalApp.ps1](./Start-LocalApp.ps1).
6. **Do not reindex** an already-populated lab volume.

The general participant workflow keeps the UI terminal open. It does not create a Windows scheduled task automatically.

If the facilitator preconfigures a login-triggered UI task, document its account, working directory, exact Python path, and restart behavior. Do not launch a second UI on the same port. Docker Desktop still requires the appropriate user session and engine startup.

## 16. Troubleshooting playbook

Use targeted logs, for example:

```powershell
docker compose -f .\compose.local-ai.yaml logs --tail 50 document layout
docker compose -f .\compose.local-ai.yaml logs --tail 50 language ner keyphrase sentiment
docker compose -f .\compose.local-ai.yaml logs --tail 50 ollama qdrant
```

Review logs for sensitive information before sharing them.

| Symptom | What to check | Safe next action |
|---|---|---|
| Docker prints only a Client section | Docker Desktop engine/user session | Start Desktop and wait for the Server section; do not continue on client-only output. |
| ARM64 host | `docker info` architecture | Use a supported x64 host; do not make an emulated ARM setup the workshop baseline. |
| Disk full or image-download I/O error | Free space on the actual Docker data volume | Stop downloads. Use Docker's supported data-location workflow or an approved disk expansion. |
| Docker stuck during data relocation | Whether the move actually completed and free space changed | Merely selecting an E: path is not evidence of a completed migration. Escalate before deleting files. |
| Missing `/etc/passwd`, `root` user, or executables after a full-disk incident | Image extraction/store integrity | Restore free space, restart, and investigate only task-owned images first. Do not assume an application bug. |
| Persistent image-store failures | Docker Desktop image-store configuration | A facilitator-approved switch from containerd to the classic store can isolate the issue. Existing store contents are retained but hidden while another store is selected; expect new downloads and account for storage. |
| `AuthenticationTypeDisabled` / HTTP 403 | Resource key-authentication policy | Stop for resource-owner/security review. Do not apply generic exclusion tags or bypass policy. |
| Billing/license failure | Correct resource, key, endpoint, eligibility, network access | Review the relevant container logs and current Microsoft container documentation. |
| NER HTTP 404 | Correct route for the NER image | This stack's NER uses `/language/:analyze-text?api-version=2023-04-01`; it does not use the older entities route. |
| Document Intelligence API mismatch | Image generation and API version | General Document v3.0 uses application API `2022-08-31` in the launcher; do not mix it blindly with another SDK/API generation. |
| Missing Document Intelligence Layout service | Compose service state/logs | Bring up both `document` and `layout`; a published Document port alone is insufficient. |
| Model list succeeds but inference fails | Run the real verifier | Listing models does not prove the runtime binaries can load them. The isolated Ollama container avoids depending on host inference binaries. |
| Models missing in the container | Actual `OLLAMA_MODELS_PATH`, manifests, and blobs | Correct the bind source. Download with a writable host store first; the container mount is read-only. |
| `QdrantClient` has no `search` | Python client version | Use the workshop's client constraint below 1.16. |
| Exact-eight validation fails | Fixtures, duplicate indexing, extraction/model versions | Inspect what changed. Do not delete an unrelated collection or reset Docker to force a pass. |
| UI says no sources | Indexing success, model choice, collection, filters | Use bge-m3, remove restrictive filters, and verify the indexed collection. |
| API status panel cannot reach port 8000 | Optional original backend is not started | The enhanced UI processes directly; a separate FastAPI backend is not required for this core lab. |
| PowerShell script blocked | Organizational script-signing/execution policy | Follow the approved script-review/signing workflow; do not globally disable policy. |
| Browser cannot open the UI from a laptop | Loopback binding and machine location | Open the UI in the Docker host's browser; do not publish ports just to work around Bastion usage. |
| VM unexpectedly stops | Auto-shutdown, automation, activity log | Confirm the shutdown owner and get approval before restarting or changing schedules. |

### Reference-lab lessons for the presenter

- An initially full OS disk prevented image extraction and stalled Docker.
- The approved recovery expanded the reference OS disk from 127 GiB to 256 GiB.
- The reference host ultimately used Docker Desktop's classic image store.
- A host Ollama installation could list models but failed real inference; an isolated official container resolved that dependency.
- Language detection and NER alone did not replace all Language features.
- An English-only scope was explicitly agreed rather than claiming unverified Arabic parity.

These are troubleshooting lessons, not instructions to resize every partner's VM or change every host's image store. Obtain approval for downtime, cost changes, and shared-host impacts.

## 17. End-of-session shutdown and cost control

Stop the foreground Streamlit process with `Ctrl+C` in its terminal.

If a facilitator-created scheduled task owns the UI, stop that specific task through its documented procedure instead.

Stop the stack while retaining containers and data:

```powershell
docker compose -f .\compose.local-ai.yaml stop
```

Or remove this Compose project's containers/network while retaining its named data volume:

```powershell
docker compose -f .\compose.local-ai.yaml down
```

Do not add `-v` unless the lab owner explicitly approves deleting that project's indexed data. Do not run broad Docker prune commands, factory reset, or delete Docker's virtual disks as routine cleanup.

The bind-mounted host model directory is read-only to the container and is not removed by `compose down`.

For Azure-hosted labs:

- Have the owner stop/deallocate the correct VM when appropriate.
- Compute billing, disks, Bastion, Azure AI usage, and other retained resources have separate cost implications.
- Deallocation does not delete disks or eliminate all charges.
- Restore any approved temporary security exceptions through the governance owner's process.
- Rotate exposed/shared lab credentials as appropriate.

## 18. Facilitator completion checklist

- [ ] Partners have the reviewed workshop release, not an incomplete default branch.
- [ ] x64 Docker Server and sufficient memory/storage were verified.
- [ ] Azure resource eligibility, licensing, billing, and governance gates were cleared.
- [ ] No real keys were shown, committed, or packaged.
- [ ] Eight containers are running with loopback-only published ports.
- [ ] Real PDF, Language, embedding, and generation checks passed.
- [ ] Two English sources produced the expected eight chunks on a fresh baseline.
- [ ] The non-English document was explicitly skipped.
- [ ] A real Streamlit query returned sources and a generated answer.
- [ ] Participants inspected grounding rather than trusting the confidence display alone.
- [ ] Participants can explain local inference versus fully disconnected operation.
- [ ] Restart, data persistence, shutdown, and ongoing costs were covered.

## 19. Official references and repository entry points

- [Document Intelligence containers](https://learn.microsoft.com/azure/ai-services/document-intelligence/containers/install-run?view=doc-intel-3.1.0)
- [Language detection containers](https://learn.microsoft.com/azure/ai-services/language-service/language-detection/how-to/use-containers)
- [Named entity recognition containers](https://learn.microsoft.com/azure/ai-services/language-service/named-entity-recognition/how-to/use-containers)
- [Key-phrase extraction containers](https://learn.microsoft.com/azure/ai-services/language-service/key-phrase-extraction/how-to/use-containers)
- [Sentiment containers](https://learn.microsoft.com/azure/ai-services/language-service/sentiment-opinion-mining/how-to/use-containers)
- [Docker Desktop Windows installation](https://docs.docker.com/desktop/setup/install/windows-install/)
- [Docker Desktop settings](https://docs.docker.com/desktop/settings-and-maintenance/settings/)
- [Ollama](https://ollama.com/)
- [Repository overview](./README.md)
- [Compose stack](./compose.local-ai.yaml)
- [Local launcher](./Start-LocalApp.ps1)
- [Real inference and RAG verifier](./tests/verify_local_services.py)

Review current product prerequisites and licensing before each delivery. Treat the tested workshop release and its validation results as the baseline, not mutable image tags alone.
