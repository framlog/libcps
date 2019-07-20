#pragma once

#include "third_party/ev.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <experimental/optional>
#include <functional>
#include <memory>
#include <queue>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <chrono>

/// \brief A future/promise model implementation backed by libev.
/// Each future is completed by the thread that ev_loop runs.

// NB:
// 1. Public methods of Future aren't thread safe and can only be
// invoked on the thread that ev_loop runs to avoid race condition.
// 2. This file trys to resolve the main drawbacks in `primitives.hpp`, which are:
//      a. Not support move-only then handlers
//      b. Two many async events
//   And it's still working in progress.

namespace cps {

#define DISABLE_COPY(type) type(const type &) = delete
#define DISABLE_MOVE(type) type(type &&) = delete
#define PIN_TYPE(type)                                                         \
  DISABLE_COPY(type);                                                          \
  DISABLE_MOVE(type);

template <typename T> using optional = std::experimental::optional<T>;

template<typename T> class Promise;
template<typename T> class FutureHolder;
template<typename Input> class ThenHolderBase;
template<typename Input, typename DoneFn, typename ErrorFn> class ThenHolder;

template<typename Input, typename Fn>
using FnRet = decltype(std::declval<Fn>()(std::declval<std::add_rvalue_reference_t<Input>>()));

enum class FutureState {
  Pending,
  Done,
  Error,
};

template<typename T>
class Event final {
public:
    Event(struct ev_loop* loop) noexcept;
    ~Event();
    PIN_TYPE(Event);

    Promise<T> Create(); 

private:
    friend class Promise<T>;
    void Notify() { ev_async_send(loop_, &async_); }
    static void OnNotify(struct ev_loop *loop, struct ev_async *w, int revent);

private:
    struct ev_loop* loop_;
    struct ev_async async_;
    std::queue<Promise<T>> pendings_;
};

template<typename T>
class Future {
public:
    using Output = T;

public:
  virtual ~Future() {}
  PIN_TYPE(Future);

  bool IsReady() const { return FutureState::Pending != state_.load(); }
  bool IsError() const { return FutureState::Error != state_.load(); }

  virtual bool SetValue(const T&);
  virtual bool SetValue(T&&);
  virtual bool SetError();

protected:
  Future() noexcept = default;
  void DoThen();

protected:
  std::atomic<FutureState> state_;
  optional<T> v_;
  std::shared_ptr<ThenHolderBase<T>> then_;
};

template<>
class Future<void> {
public:
    using Output = void;

public:
  virtual ~Future() {}
  PIN_TYPE(Future);

  bool IsReady() const { return FutureState::Pending != state_.load(); }
  bool IsError() const { return FutureState::Error != state_.load(); }

  virtual bool SetValue();
  virtual bool SetError();
  void DoThen();

protected:
  Future() noexcept = default;

protected:
  std::atomic<FutureState> state_;
  std::shared_ptr<ThenHolderBase<void>> then_;
};

template<typename T>
class FutureHolder: public Future<T> {
public:
    PIN_TYPE(FutureHolder);

private:
    friend class Event<T>;
    FutureHolder() = default;
};

template<>
class FutureHolder<void>: public Future<void> {
public:
    PIN_TYPE(FutureHolder);

private:
    friend class Event<void>;
    FutureHolder() = default;
};

template<typename Input>
class ThenHolderBase {
private:
    friend class Future<Input>;
    virtual void OnThen(Input&& v) = 0;
    virtual void OnError() = 0;
};

template<>
class ThenHolderBase<void> {
private:
    friend class Future<void>;
    virtual void OnThen() = 0;
    virtual void OnError() = 0;
};

template<typename Input, typename ThenFn, typename ErrorFn>
class ThenHolder final: public Future<FnRet<Input, ThenFn>>, public ThenHolderBase<Input> {
public:
    PIN_TYPE(ThenHolder);

private:
    void OnThen(Input&& v) override;
    void OnError() override;

private:
    ThenFn on_then_;
    ErrorFn on_error_;
};

template<typename ThenFn, typename ErrorFn>
class ThenHolder<void, ThenFn, ErrorFn> final: public Future<FnRet<void, ThenFn>>, public ThenHolderBase<void> {
public:
    PIN_TYPE(ThenHolder);

private:
    void OnThen() override;
    void OnError() override;

private:
    ThenFn on_then_;
    ErrorFn on_error_;
};



template<typename T>
class Promise {
public:
    std::shared_ptr<FutureHolder<T>> GetFuture() { return fu_; }
    void SetValue(const T& value);
    void SetValue(T&& value);
    void SetError();

private:
    friend class Event<T>;
    Promise(Event<T>* event_) noexcept;

private:
    std::shared_ptr<FutureHolder<T>> fu_;
    Event<T>* const event_;
};

template<>
class Promise<void> {
public:
    std::shared_ptr<FutureHolder<void>> GetFuture() { return fu_; }
    void SetValue();
    void SetError();

private:
    friend class Event<void>;
    Promise(Event<void>* event_) noexcept;

private:
    std::shared_ptr<FutureHolder<void>> fu_;
    Event<void>* const event_;
};

// Implementations

template<typename T>
Event<T>::Event(struct ev_loop* loop) noexcept: loop_(loop) {
    ev_async_init(&async_, OnNotify);
    async_.data = this;
    ev_async_start(loop, &async_);
}

template<typename T>
Event<T>::~Event() {
    ev_async_stop(loop_, &async_);
}

template<typename T>
Promise<T> Event<T>::Create() {
    pendings_.emplace(this);
    return pendings_.back();
}

template<typename T>
void Event<T>::OnNotify(struct ev_loop *loop, struct ev_async *w, int revent) {
    auto e = reinterpret_cast<Event<T> *>(w->data);
    while (e->pendings_.size() > 0 && e->pendings_.front()->IsReady()) {
        e->pendings_.front()->DoThen();
        e->pendings_.pop();
    }
}

template<typename T>
bool Future<T>::SetValue(const T& v) {
    FutureState expect = FutureState::Pending;
    if (state_.compare_exchange_strong(expect, FutureState::Done)) {
        v_.emplace(v);
        return true;
    } else {
        return false;
    }
}

template<typename T>
bool Future<T>::SetValue(T&& v) {
    FutureState expect = FutureState::Pending;
    if (state_.compare_exchange_strong(expect, FutureState::Done)) {
        v_.emplace(std::move(v));
        return true;
    } else {
        return false;
    }
}

template<typename T>
bool Future<T>::SetError() {
    FutureState expect = FutureState::Pending;
    return state_.compare_exchange_strong(expect, FutureState::Error);
}

bool Future<void>::SetValue() {
    FutureState expect = FutureState::Pending;
    return state_.compare_exchange_strong(expect, FutureState::Done);
}

template<typename T>
void Future<T>::DoThen() {
    if (then_) {
        switch (state_.load()) {
        case FutureState::Done:
            then_->OnThen(std::move(v_.value()));
        case FutureState::Error:
            then_->OnError();
        default:
            assert(false);
        }
    }
}

void Future<void>::DoThen() {
    if (then_) {
        switch (state_.load()) {
        case FutureState::Done:
            then_->OnThen();
        case FutureState::Error:
            then_->OnError();
        default:
            assert(false);
        }
    }
}

template<typename Input, typename ThenFn, typename ErrorFn>
void ThenHolder<Input, ThenFn, ErrorFn>::OnThen(Input&& v) {
    try {
        SetValue(on_then_(std::move(v)));
    } catch (std::exception& e) {
        on_error_();
        SetError();
    }
    DoThen();
}

template<typename Input, typename ThenFn, typename ErrorFn>
void ThenHolder<Input, ThenFn, ErrorFn>::OnError() {
    on_error_();
    SetError();
    DoThen();
}

template<typename ThenFn, typename ErrorFn>
void ThenHolder<void, ThenFn, ErrorFn>::OnThen() {
    try {
        SetValue(on_then_());
    } catch (std::exception& e) {
        on_error_();
        SetError();
    }
    DoThen();
}

template<typename T>
Promise<T>::Promise(Event<T>* event) noexcept: fu_(new FutureHolder<T>()), event_(event) {}

template<typename T>
void Promise<T>::SetValue(const T& value) {
    if (fu_->SetValue(value)) {
        event_->Notify();
    }
}

template<typename T>
void Promise<T>::SetValue(T&& value) {
    if (fu_->SetValue(std::move(value))) {
        event_->Notify();
    }
}

template<typename T>
void Promise<T>::SetError() {
    if (fu_->SetError()) {
        event_->Notify();
    }
}

void Promise<void>::SetValue() {
    if (fu_->SetValue()) {
        event_->Notify();
    }
}

} // namespace cps
