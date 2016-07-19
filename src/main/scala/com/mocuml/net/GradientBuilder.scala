package com.mocuml.net

trait GradBuilder {
    def buildGrad(delta : DMD)

    def biasGrad(delta : DMD)

    def weightGrad(delta : DMD)
}

