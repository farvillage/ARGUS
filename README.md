# ARGUS

**A pre-sign security linter for unsigned EVM transactions.**

`ARGUS` inspects a transaction *before* a wallet signs it and flags
wallet drainers, permit phishing, EIP-7702 delegation hijacks, dangerous
bytecode, and asymmetric asset drains. It runs as a CLI, in CI (SARIF
for GitHub code scanning), or from any IDE using `python3 -m argus`. It
also works fully offline against fixtures, with no RPC key.

## Why

Most drained wallets were not hacked. The owner signed something they
could not read. Calldata is opaque, permits are off-chain, and since
EIP-7702, a single signature can hand an EOA's code execution to an
arbitrary contract.

`ARGUS` is a deterministic checkpoint between **building the payload**
and **signing it**.

## Architecture

``` mermaid
flowchart LR
    P[Unsigned tx JSON<br/>from, to, data, value,<br/>chainId, authorizationList] --> L[load_payload<br/>strict Pydantic v2]
    L --> D[decode_calldata<br/>eth-abi]
    D --> S{Simulator}
    S -->|--offline| O[OfflineFixtureSimulator<br/>_mock_context, zero network]
    S -->|--rpc-url| R[LiveRPCSimulator<br/>CCIP-Read disabled<br/>eth_getCode / eth_call /<br/>debug_traceCall]
    O --> C[SimulationContext]
    R --> C
    B[disassemble_opcodes<br/>pure Python, PUSH-aware] --> C
    C --> E[Rule engine<br/>ARG001 - ARG006]
    E --> RP[LintReport<br/>findings + risk score]
    C -.->|--explain| M[ml_scorer<br/>features -> ONNX model -> SHAP]
    M -.-> RP
    RP --> F[Formatters<br/>rich / json / SARIF 2.1.0]
    RP --> X[Exit code<br/>1 if finding >= --fail-on]
```

## Quickstart

The following setup works in any IDE and does not require adding a
command to your system `PATH`.

``` bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # Core dependencies, pytest, and Ruff
pip install -e ".[ml]"               # Optional: ONNX and SHAP explanations

python3 -m argus rules
python3 -m argus check examples/permit2_phishing.json --offline
python3 -m argus check examples/safe_uniswap_swap.json --offline
python3 -m argus check examples/proxy_delegatecall_drainer.json --offline --explain
```

Once installed, commands can also be run using `argus ...`. In an IDE,
run `python3 -m argus` as a module. Run the test suite with `pytest` or
`python3 -m pytest`.

## Usage

``` text
argus check <payload.json> [--rpc-url URL] [--offline] [--explain]
             [--format rich|json|sarif] [--fail-on low|medium|high|critical]
argus rules
```

  -----------------------------------------------------------------------
  Exit code                           Meaning
  ----------------------------------- -----------------------------------
  `0`                                 No finding at or above `--fail-on`
                                      (default: `high`).

  `1`                                 At least one finding at or above
                                      `--fail-on`.

  `2`                                 Operational error, such as an
                                      invalid payload, no backend
                                      selected, or an RPC error.
  -----------------------------------------------------------------------

The RPC URL can also be supplied through the `ARGUS_RPC_URL` environment
variable.

## Payload format

The payload is a flat JSON object containing unsigned-transaction
fields. Quantity values accept integers or hexadecimal strings prefixed
with `0x`.

``` json
{
  "chainId": 1,
  "from": "0x...",
  "to": "0x...",
  "value": "0x0",
  "data": "0x...",
  "authorizationList": [
    {
      "chainId": "0x0",
      "address": "0x...",
      "nonce": "0x1"
    }
  ],
  "_mock_context": {
    "is_contract": true,
    "is_verified": false,
    "deployment_age_blocks": 37
  }
}
```

`_mock_context` supplies data to the offline simulator. It can contain
target metadata, per-address metadata, simulated state diffs, and
optional `runtime_bytecode`, which is disassembled by the pure-Python
disassembler.

## Detection rules

  ----------------------------------------------------------------------------
  Rule ID                 Severity                Detects
  ----------------------- ----------------------- ----------------------------
  `ARG001`                HIGH / CRITICAL         Unbounded `approve`
                                                  (`>= 2^255`) or
                                                  `setApprovalForAll(true)`.
                                                  Severity is CRITICAL for an
                                                  EOA or unverified spender.

  `ARG002`                CRITICAL                EIP-2612 `permit` to an EOA,
                                                  or a deadline more than 30
                                                  days in the future.

  `ARG003`                CRITICAL                EIP-7702 delegation to an
                                                  unverified or unknown
                                                  contract; also flags
                                                  `chain_id = 0`
                                                  replayability.

  `ARG004`                HIGH                    Target deployed fewer than
                                                  500 blocks ago and not
                                                  verified.

  `ARG005`                HIGH                    Non-standard `DELEGATECALL`
                                                  (without an EIP-1967 slot)
                                                  or `SELFDESTRUCT` in
                                                  unverified bytecode.

  `ARG006`                CRITICAL                Net outflow of ETH/ERC-20
                                                  assets without inbound
                                                  assets or receipt tokens.
  ----------------------------------------------------------------------------

The deterministic risk score is the sum of severity weights, capped at
100:

  Severity     Weight
  ---------- --------
  LOW               5
  MEDIUM           15
  HIGH             35
  CRITICAL         60

## Sample output

``` text
$ python3 -m argus check examples/permit2_phishing.json --offline
╭──────────────────────────── Decoded Call ─────────────────────────────╮
│ From      0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc                  │
│ To        0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48                  │
│ Function  permit                                                      │
│   spender 0xbadc0de0badc0de0badc0de0badc0de0badc0de0                  │
│   value   1157920892373161954235709850086879078532699846656405640394… │
╰───────────────────────────────────────────────────────────────────────╯
Rule    Severity  Title                               Details / Remediation
ARG002  CRITICAL  EIP-2612 permit phishing signature  Permit signature looks like phishing:
                                                      spender is an EOA; deadline is more
                                                      than 30 days ahead.
                                                      Fix: Do not sign. ...
╭─────────────────────────────── Verdict ────────────────────────────────╮
│          DO NOT SIGN - deterministic risk score 60/100                │
╰───────────────────────────────────────────────────────────────────────╯
$ echo $?
1
```

## Explainable AI scorer and real-data training

The detection rules are deterministic and auditable. The optional
`--explain` flag adds an independent machine-learning assessment.

### Feature extraction

`extract_features` derives eight behavioral features from the
simulation:

1.  Unbounded approval.
2.  Untrusted spender.
3.  Permit deadline horizon.
4.  Untrusted EIP-7702 delegate.
5.  Unverified target.
6.  Contract freshness.
7.  Dangerous opcodes.
8.  Asymmetric outflow.

### Multi-model cross-validation and ONNX export

`ARGUS` supports benchmarking the following models using Stratified
K-Fold cross-validation:

-   Random Forest (`rf`)
-   HistGradientBoosting (`hist_gbdt`)
-   XGBoost (`xgboost`)

The best-performing model is exported to ONNX for CPU inference, with a
target inference time below 5 ms.

### Real-world incident harvesting

The training pipeline can use historical exploit and drainer transaction
hashes parsed from `DeFi-Hack-Chronicle`, combined with mainnet
user-traffic data. This process requires access to an RPC endpoint and
an Etherscan V2 API key.

#### Configure credentials

Create a local `.env` file in the project root:

``` ini
ETHERSCAN_API_KEY="your_etherscan_key_here"
ARGUS_RPC_URL="https://ethereum-rpc.publicnode.com"
```

#### Run the harvesting and training pipeline

``` bash
# 1. Clone DeFi-Hack-Chronicle and parse seed transactions
git clone --depth 1 https://github.com/DeFiHackLabs/DeFi-Hack-Chronicle.git /tmp/DeFi-Hack-Chronicle
python3 scripts/parse_chronicle.py --chronicle-dir /tmp/DeFi-Hack-Chronicle

# 2. Harvest real mainnet transactions
python3 scripts/harvest_real_data.py --seeds data/chronicle_seeds.json --max-malicious 90

# 3. Benchmark models, select the winner, and export it to ONNX
python3 scripts/train_model.py --dataset data/real_transactions.csv --model auto
```

## GitHub code scanning (SARIF)

The CLI can generate SARIF output for GitHub code scanning. Add the
following steps to a GitHub Actions workflow:

``` yaml
- name: Run ARGUS
  run: python3 -m argus check tx/payload.json --offline --format sarif > argus.sarif || true

- name: Upload SARIF results
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: argus.sarif
```

## Testing

Run the test suite with:

``` bash
pytest -q
```

The suite covers every rule (`ARG001`--`ARG006`), including boundary
conditions, PUSH-skipping disassembly, EIP-1967 pairing, calldata
decoding, SARIF structure, CLI exit codes, CCIP-Read hardening, and the
ONNX/SHAP pipeline.

## Security hardening

### Pre-sign isolation

`ARGUS` runs before the signer and:

-   Never holds keys and never signs, broadcasts, or sends transactions.
-   Uses only read-only RPC calls: `eth_getBlockByNumber`,
    `eth_getCode`, `eth_call`, and `debug_traceCall`.
-   Performs zero network I/O when running with `--offline`.
-   Avoids pickle deserialization. The ML model uses ONNX, while SHAP
    explanations are reconstructed deterministically from compressed
    NumPy arrays or safe XGBoost JSON sidecars.

### OffchainLookup / CCIP-Read SSRF mitigation

`LiveRPCSimulator` disables CCIP-Read globally using
`global_ccip_read_enabled = False` and passes `ccip_read_enabled=False`
to every `eth_call`. This is intended to prevent malicious contracts
from triggering SSRF requests through EIP-3668.

## Limitations

-   Source-code verification and contract age checks require explorer or
    archive data. Offline fixtures provide this information; live mode
    checks `ARGUS_VERIFIED_ADDRESSES`.
-   `debug_traceCall` is required for full state diffs.
-   A clean report is not a guarantee of safety. `ARGUS` reduces risk
    but is not a substitute for a formal audit.

## License

MIT

------------------------------------------------------------------------

## Save and commit the README

Save this content as `README.md`, then commit and push the update:

``` bash
git add README.md
git commit -m "docs: update README with real-world DeFi-Hack-Chronicle harvesting and multi-model benchmark"
git push
```
