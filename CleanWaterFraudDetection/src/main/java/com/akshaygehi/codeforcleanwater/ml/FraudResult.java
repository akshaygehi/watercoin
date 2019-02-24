package com.akshaygehi.codeforcleanwater.ml;

import java.io.Serializable;

public class FraudResult implements Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	
	private final long totalFrauds;
    private final long foundFrauds;
    private final long flaggedFrauds;

    public FraudResult(long totalFrauds, long foundFrauds, long flaggedFrauds) {
        this.totalFrauds = totalFrauds;
        this.foundFrauds = foundFrauds;
        this.flaggedFrauds = flaggedFrauds;
    }

    public long getTotalFrauds() {
        return totalFrauds;
    }

    public long getFoundFrauds() {
        return foundFrauds;
    }

    public long getFlaggedFrauds() {
        return flaggedFrauds;
    }

}