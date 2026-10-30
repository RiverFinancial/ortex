FROM hexpm/elixir:1.20.4-erlang-29.1.1-alpine-3.24.2@sha256:ad851f40ce103dcb4ad56f23877d99473ef5c013e09b9e57921ffe877ac6d6a9 AS build
RUN apk add --no-cache build-base git bash python3 rust=1.96.1-r0 cargo=1.96.1-r0 ca-certificates && \
    apk add --no-cache \
      --repository=https://dl-cdn.alpinelinux.org/alpine/edge/main \
      --repository=https://dl-cdn.alpinelinux.org/alpine/edge/community \
      --repository=https://dl-cdn.alpinelinux.org/alpine/edge/testing \
      onnxruntime=1.24.4-r1
WORKDIR /app
ENV MIX_ENV=test ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so.1.24.4 RUSTUP_TOOLCHAIN=1.96.1
COPY mix.exs mix.lock ./
RUN mix local.hex --force && mix local.rebar --force && mix deps.get --check-locked
COPY . .
RUN mix compile --warnings-as-errors
RUN mix test && python3 scripts/fixtures.py && mix test test/models --include model && \
    python3 scripts/scenario.py && python3 scripts/package.py test --mode source

FROM hexpm/elixir:1.20.4-erlang-29.1.1-alpine-3.24.2@sha256:ad851f40ce103dcb4ad56f23877d99473ef5c013e09b9e57921ffe877ac6d6a9
RUN apk add --no-cache git bash python3 ca-certificates && \
    apk add --no-cache \
      --repository=https://dl-cdn.alpinelinux.org/alpine/edge/main \
      --repository=https://dl-cdn.alpinelinux.org/alpine/edge/community \
      --repository=https://dl-cdn.alpinelinux.org/alpine/edge/testing \
      onnxruntime=1.24.4-r1
WORKDIR /app
ENV MIX_ENV=test ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so.1.24.4
COPY --from=build /app /app
RUN ! command -v cargo && ! command -v rustc
CMD ["sh", "-ec", "python3 scripts/package.py test --mode prebuilt && if test -n \"${TARGET:-}\"; then python3 scripts/package.py nif --target \"$TARGET\"; fi"]
