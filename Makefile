.PHONY: help
.DEFAULT_GOAL := help

CONTAINER_NAME := stream-vigil

shell: ## Login shell
	@docker exec -it $(CONTAINER_NAME) bash

fmt: ## Format python code
	@rye run fmt

lint: ## Lint python code
	@rye run lint

test: ## Run test
	@rye run test

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
