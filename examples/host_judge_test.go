package main

import (
	"testing"

	"github.com/caimlas/classi-fly/reservoir"
)

// result is a helper that constructs reservoir.Result values directly, so
// these tests never need a real .fly artifact on disk.
func result(class string, conf, margin float64, abstained bool) reservoir.Result {
	return reservoir.Result{Class: class, Confidence: conf, Margin: margin, Abstained: abstained}
}

func TestJudgeDecide(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		hostLabel string
		hostConf  float64
		res       reservoir.Result
		wantRoute bool
		wantLabel string
	}{
		{"agree above threshold routes", "coding", 0.90, result("coding", 0.85, 0.4, false), true, "coding"},
		{"exactly at threshold routes", "git", 0.70, result("git", 0.99, 0.9, false), true, "git"},
		{"host below threshold falls through", "coding", 0.69, result("coding", 0.99, 0.5, false), false, ""},
		{"reservoir disagrees falls through", "coding", 0.95, result("debugging", 0.98, 0.9, false), false, ""},
		{"reservoir abstained falls through", "coding", 0.95, result("", 0, 0, true), false, ""},
		{"both unsure falls through", "chat", 0.50, result("chat", 0.10, 0.01, false), false, ""},
		{"host disagree reservoir agrees still needs host", "review", 0.40, result("review", 0.80, 0.3, false), false, ""},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			route, label := judgeDecide(tc.hostLabel, tc.hostConf, tc.res)
			if route != tc.wantRoute {
				t.Errorf("route: got %v, want %v (host=%q conf=%.2f res=%+v)", route, tc.wantRoute, tc.hostLabel, tc.hostConf, tc.res)
			}
			if label != tc.wantLabel {
				t.Errorf("label: got %q, want %q", label, tc.wantLabel)
			}
		})
	}
}

// TestJudgeDecideNeverEscalates pins the key property: the reservoir can only
// downgrade a host decision to fall-through, never upgrade a weak host
// decision to a route, and never substitutes its own class.
func TestJudgeDecideNeverEscalates(t *testing.T) {
	t.Parallel()
	// Reservoir very confident in a different class while the host is unsure:
	// still a fall-through with no label, not a route on the reservoir's pick.
	route, label := judgeDecide("coding", 0.42, result("debugging", 0.97, 0.9, false))
	if route || label != "" {
		t.Errorf("confident reservoir must not rescue an unsure host: route=%v label=%q", route, label)
	}
	// Reservoir abstained while host is very confident: fall through.
	route, label = judgeDecide("search", 0.99, result("", 0, 0, true))
	if route || label != "" {
		t.Errorf("abstained reservoir must fall through: route=%v label=%q", route, label)
	}
}

func TestJudgeDecideEmptyHostLabel(t *testing.T) {
	t.Parallel()
	// A host head with no opinion cannot route even on agreement, because
	// the agreed label would be empty.
	route, label := judgeDecide("", 0.99, result("", 0.9, 0.4, false))
	if route || label != "" {
		t.Errorf("empty host label must never route: route=%v label=%q", route, label)
	}
}
