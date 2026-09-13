# semdoc 30-second demo: template init → ingest → search → health
Output "/workspace/semdoc/docs/demo.gif"
Set FontSize 20
Set Width 1100
Set Height 620
Set Theme { "background": "#14141f", "foreground": "#e8e8f0", "cursor": "#9ad", "black": "#14141f", "green": "#7fd88f", "yellow": "#f5d76e", "blue": "#6fb3ff", "magenta": "#c792ea", "cyan": "#7fdbff" }
Set Padding 24
Set Shell bash
Type "export SEMDOC_CONFIG=/home/vhsuser/demo-work/vhs_config.toml RUST_LOG=off"
Enter
Sleep 200ms
Type "semdoc init --template code-search --db ./demo && echo '--- template ready ---'"
Enter
Sleep 800ms
Type "semdoc add --db ./demo --file /home/vhsuser/demo-work/vhs-demo-doc1.md --source docs/config-parsing.md --meta language=rust --meta repo=acme"
Enter
Sleep 1.6s
Type "semdoc add --db ./demo --file /home/vhsuser/demo-work/vhs-demo-doc2.md --source docs/retry-policy.md --meta language=rust --meta repo=acme"
Enter
Sleep 1.6s
Type "semdoc query --db ./demo -t 'how does config parsing merge layers' --limit 1"
Enter
Sleep 2.5s
Type "clear && semdoc query --db ./demo -t 'retry backoff' --mode fts --limit 1"
Enter
Sleep 1.2s
Type "clear && semdoc doctor --db ./demo"
Enter
Sleep 1.5s
Type "clear && semdoc stats --db ./demo"
Enter
Sleep 1.5s
Ctrl+C
