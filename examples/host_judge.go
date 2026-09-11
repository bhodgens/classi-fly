// Command host_judge is a worked example of the "second-opinion judge"
// pattern: a host application already has its own classifier head and wants
// classi-fly as an independent check. The host routes a request only when its
// own head is confident AND the reservoir agrees AND the reservoir did not
// abstain. See docs/INTEGRATION.md for the full write-up.
//
// This file is deliberately generic: it imports only the reservoir package
// and names no specific consumer product.
package main

import (
	"fmt"
	"log"

	"github.com/caimlas/classi-fly/reservoir"
)

// hostThreshold is the host head's own confidence floor. Only a host decision
// above this floor is even eligible for routing; everything below falls
// through to the host's slow path regardless of what the reservoir says.
const hostThreshold = 0.70

// judgeDecide is the entire integration decision. route is true only when all
// three conditions hold:
//
//  1. the host head is above its own confidence threshold,
//  2. the reservoir did not abstain, and
//  3. the reservoir's class equals the host's label.
//
// When route is true, label is the agreed label; otherwise label is "" and
// the caller falls through to its normal (slower or interactive) path. The
// reservoir never overrides the host upward: disagreement always degrades to
// "fall through", never to "use the reservoir's answer".
//
// An empty host label (the host head produced no decision) can never satisfy
// agreement: ""=="" would be true, so it is rejected explicitly.
func judgeDecide(hostLabel string, hostConf float64, res reservoir.Result) (route bool, label string) {
	if hostLabel == "" {
		return false, ""
	}
	if hostConf < hostThreshold {
		return false, ""
	}
	if res.Abstained || res.Class != hostLabel {
		return false, ""
	}
	return true, hostLabel
}

// judgeEmbedding shows the in-process face end to end: load once at startup
// (fail-closed, but the host keeps working without the feature if the
// artifact is missing), then consult the reservoir per request.
type judgeEmbedding struct {
	res *reservoir.Reservoir
}

// newJudgeEmbedding loads the artifact. A missing or corrupt .fly returns an
// error and the caller treats the judge feature as disabled; it is a no-op
// enhancement, not a dependency.
func newJudgeEmbedding(path string) (*judgeEmbedding, error) {
	res, err := reservoir.Load(path)
	if err != nil {
		return nil, err
	}
	return &judgeEmbedding{res: res}, nil
}

// decide consults the reservoir on an embedding the host head has already
// labeled. Dim mismatch or a disabled feature returns route=false (fail
// open to the host's own path: the host remains fully functional).
func (j *judgeEmbedding) decide(hostLabel string, hostConf float64, embedding []float32) (bool, string) {
	if j == nil || j.res == nil {
		return false, ""
	}
	res, err := j.res.Classify(embedding)
	if err != nil {
		return false, ""
	}
	return judgeDecide(hostLabel, hostConf, res)
}

func main() {
	judge, err := newJudgeEmbedding("model.fly")
	if err != nil {
		// The feature is optional: without an artifact the host runs its own
		// head unchanged.
		log.Printf("second-opinion judge disabled (%v); host head only", err)
		return
	}
	info := judge.res.Info()
	fmt.Printf("judge loaded: %s (%d classes, embed_dim %d, license %s)\n",
		info.Name, len(info.Classes), info.EmbedDim, info.License)
}
