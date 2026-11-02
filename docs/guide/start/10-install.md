# Install, and see what this machine can run

You need Python 3.11 or newer. Nothing else: the sizing half of onpar has zero
runtime dependencies, so the first command runs before anything is installed.

```bash
uvx --from onpar onpar fit
```

That reads the machine it is on and prints every catalogue model that fits at
the default context and concurrency, with the arithmetic one flag away. The
`--from onpar` is not optional: the distribution and the command share a name,
and `uvx` needs to be told which is which.

## Put it on your PATH

```bash
uv tool install onpar          # or: pipx install onpar, pip install onpar
brew install dshakes/tap/onpar # macOS, from the shared tap
npx onpar-cli fit              # the npm shim execs uvx --from onpar underneath
```

`fit` is one package. Capturing and scoring your own traffic needs two more,
the capture gateway and the compiled extension that reads captures back:

```bash
pip install onpar onpar-gateway onpar-core
```

## Ask the machine what it is ready for

`onpar doctor` checks the pieces in the order the loop needs them and ends with
the one command to run next. This is a real run:

<!-- run: doctor -->

Every stage after this one refuses to start without the piece it needs, and
says which. You will not find out at the end that something was missing.

## Pin a build

The commands above fetch the newest release. When a build has to stay put:

```bash
uvx --from onpar==1.4.0 onpar fit
npx onpar-cli@1.4.0 fit
```

`onpar version` prints the installed version from its metadata, and
`onpar upgrade` prints the upgrade command for the way you installed it.

Next: [ten seconds to a fit](20-ten-seconds.md), or straight to
[the walkthrough](30-walkthrough.md).
