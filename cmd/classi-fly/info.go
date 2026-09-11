package main

import (
	"flag"
	"fmt"
	"io"
)

// runInfo implements `classi-fly info`: print the artifact's header JSON
// verbatim to stdout.
func runInfo(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	fs.SetOutput(stderr)
	allowNC := fs.Bool("allow-noncommercial", false, "permit non-commercial artifacts")
	if err := fs.Parse(reorderFlagSet(fs, args)); err != nil {
		return exitUsage
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "classi-fly info: exactly one .fly path is required")
		return exitUsage
	}
	raw, header, err := readFlyHeader(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "classi-fly info: %v\n", err)
		return exitLoad
	}
	if !*allowNC && isNonCommercial(header.License) {
		fmt.Fprintln(stderr, "classi-fly info: non-commercial artifact refused; re-run with --allow-noncommercial to override")
		return exitLicense
	}
	if _, err := fmt.Fprintf(stdout, "%s\n", raw); err != nil {
		fmt.Fprintf(stderr, "classi-fly info: writing output: %v\n", err)
		return exitLoad
	}
	return exitOK
}
