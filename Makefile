## ----------------------------------------------------------------------
## The purpose of this Makefile is to abstract common commands for
## building and running the risknet application.
## ----------------------------------------------------------------------


help:                       ## show this help
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)


# docker commands

build-image:                ## build docker image
	docker build -t risknet:latest . --build-arg gcp_project=${PROJECT}

it-shell:                   ## run interactive shell in docker container
	docker run --mount type=bind,source=$(shell pwd)/secrets,target=/secrets -it homebase bash

push-image:		    ## push image to GCR
	docker tag risknet gcr.io/${PROJECT}/risknet
	docker push gcr.io/${PROJECT}/risknet

# k8s commands

get-gke-cred:	            ## get GKE credentials (if applicable)
	 gcloud container clusters get-credentials $(cluster) --region $(region)


start-k8s-local:            ## start local k8s via minikube
	minikube start --driver=hyperkit --memory 8192 --cpus 4

verify-k8s-dns:             ## verify that k8s dns is working properly
	sleep 10
	kubectl apply -f https://k8s.io/examples/admin/dns/dnsutils.yaml
	sleep 20
	kubectl exec -i -t dnsutils -- nslookup kubernetes.default
	kubectl delete -f https://k8s.io/examples/admin/dns/dnsutils.yaml

patch-container-registry:   ## patch cluster to point to private repository - usually necessary for Minikube
	kubectl --namespace=spark-operator create secret docker-registry gcr-json-key \
			  --docker-server=https://gcr.io \
			  --docker-username=_json_key \
			  --docker-password="$$(cat secrets/key-file)" \
			  --docker-email=${KUBEUSER}@${KUBEDOMAIN}

	kubectl --namespace=spark-operator patch serviceaccount my-release-spark \
			  -p '{"imagePullSecrets": [{"name": "gcr-json-key"}]}'

run-job:		    ## run spark job via k8s manifest with injected environment variables
ifdef manifest
	envsubst < $(manifest) | kubectl apply -f -
else
	@echo 'No manifest defined. Indicate as follows: *make manifest=manifest/job.yaml run-job*'
endif

# python

create-activate-venv:       ## make and activate python virtual environment
	python3 -m venv env
	echo "Now run: source env/bin/activate. Finally run: pip install build"

build:                      ## build python tarball and wheel
	python${PYTHON_VERSION} -m build

install:                    ## install python wheel
	pip${PYTHON_VERSION} install dist/risknet-*.whl --no-cache-dir --force-reinstall

clean-install: clean build  ## clean artifacts and install install python wheel
	pip${PYTHON_VERSION} install dist/risknet-*.whl --no-cache-dir --force-reinstall

clean:                      ## clean artifacts
	rm -r -f dist*
	rm -r -f src/*.egg-info
	rm -r -f .mypy_cache

check_types:                ## run mypy type checker
	mypy src/risknet

lint:                       ## run flake8 linter
	flake8 src/risknet

analyze: check_types lint   ## run full code analysis

test:			    ## run tests locally
	coverage run -m pytest

docker-test: build-image    ## run tests in docker
	docker run risknet make test

