SHELL := /bin/bash

.PHONY: all run test open-notebooks

all: run test

run:
	./run_all.sh

test:
	dotnet test lab1/tests/Lab1.Tests/Lab1.Tests.csproj
	dotnet test lab2/tests/Lab2.Tests/Lab2.Tests.csproj

open-notebooks:
	./run_all.sh --open-notebooks
