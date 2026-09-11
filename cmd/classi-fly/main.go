// Command classi-fly packs, inspects, and runs fly-reservoir classifiers.
//
// Subcommands:
//
//	pack      assemble a .fly artifact from an adjacency + readout export
//	info      print a .fly artifact's header JSON verbatim
//	classify  classify one embedding with a .fly artifact
//	build     convenience: synthetic reservoir + training + pack in one step
package main

import (
	"fmt"
	"os"
)

// CLI exit codes (frozen by the CLI contract).
const (
	exitOK      = 0 // success
	exitUsage   = 1 // usage error: bad flags, missing arguments, missing tools
	exitLoad    = 2 // load/validation error: unreadable or inconsistent artifact/embedding
	exitLicense = 3 // non-commercial artifact refused without --allow-noncommercial
)

const usageText = `classi-fly - fly-reservoir classifier CLI

Usage:

  classi-fly pack --adjacency a.json --readout r.json --pack p.json --out m.fly
                  [--decay 0.8] [--allow-noncommercial]
  classi-fly info m.fly
  classi-fly classify m.fly --embedding 0.1,0.2
  classi-fly classify m.fly --embedding-file vec.json
  classi-fly build --synthetic --classes c.json --pairs p.jsonl --out m.fly
                   [--seed 7] [--neurons 2048] [--steps 8] [--decay 0.8]
                   [--python python3] [--tools-dir .]

Exit codes:
  0 success
  1 usage error
  2 artifact/embedding load or validation error
  3 non-commercial artifact refused without --allow-noncommercial
`

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usageText)
		os.Exit(exitUsage)
	}
	var code int
	switch args[0] {
	case "pack":
		code = runPack(args[1:], os.Stdout, os.Stderr)
	case "info":
		code = runInfo(args[1:], os.Stdout, os.Stderr)
	case "classify":
		code = runClassify(args[1:], os.Stdout, os.Stderr)
	case "build":
		code = runBuild(args[1:], os.Stdout, os.Stderr)
	case "serve":
		code = runServe(args[1:])
	case "help", "-h", "--help":
		fmt.Print(usageText)
		code = exitOK
	default:
		fmt.Fprintf(os.Stderr, "classi-fly: unknown subcommand %q\n\n%s", args[0], usageText)
		code = exitUsage
	}
	os.Exit(code)
}
