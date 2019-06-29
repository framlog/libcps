#pragma once

#include "third_party/ev.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <experimental/optional>
#include <functional>
#include <list>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <chrono>

#ifdef LOG_EXCEPTION
#include <iostream>
#endif

/// \brief A future/promise model implementation backed by libev.
/// Each future is completed by the thread that ev_loop runs.
/// A future lives until it's done or get error at least if
/// the caller don't cancel it.

// NB:
// 1. Public methods of Future aren't thread safe and can only be
// invoked on the thread that ev_loop runs to avoid race condition.

namespace cps {

#define DISABLE_COPY(type) type(const type &) = delete
#define DISABLE_MOVE(type) type(type &&) = delete
#define PIN_TYPE(type)                                                         \
  DISABLE_COPY(type);                                                          \
  DISABLE_MOVE(type);

template <typename T> using optional = std::experimental::optional<T>;

class AsyncNotifier;
using FutureContext = AsyncNotifier;
template <typename T> class Future;
template <typename T> class SharedFuture;
template <typename T, template<typename> class FutureT = Future> class Promise;
template <typename... Futures> class WhenAll;
template <typename T> class WhenEach;
template <typename... Futures>
std::shared_ptr<WhenAll<Futures...>>
MakeWhenAll(std::shared_ptr<FutureContext> ctx, Futures &&... futures);
template <typename T>
std::shared_ptr<WhenEach<T>> MakeWhenEach(std::shared_ptr<FutureContext> ctx);
template <std::size_t Index, typename Handle, typename Future,
          typename... LeftFutures>
void FutureIterate(std::add_pointer_t<Handle> handle, Future &&future,
                   LeftFutures &&... lefts);

enum class FutureState {
  Pending,
  Done,
  Error,
};

class Futurable {
public:
  virtual ~Futurable() {}

  virtual bool IsReady() const = 0;
  virtual bool IsError() const = 0;
  virtual void GetNotified() = 0;

protected:
  Futurable() noexcept = default;
};

class AsyncNotifier final {
public:
  AsyncNotifier(struct ev_loop *loop) noexcept : loop_(loop), last_freezed_size_(0) {
    ev_async_init(&async_, OnNotify);
    async_.data = this;
    ev_async_start(loop, &async_);
  }
  ~AsyncNotifier() { ev_async_stop(loop_, &async_); }
  PIN_TYPE(AsyncNotifier);

  // NB: This method must be invoked on the thread that loop_ runs.
  void Chained(Futurable *future) { futures_.push_back(future); }

  void FreezeOrder() { last_freezed_size_ = futures_.size(); }

  void RunTimer(struct ev_timer *timer) {
      ev_timer_start(loop_, timer);
  }

  void Remove(Futurable *future) {
    for (auto iter = futures_.begin(); iter != futures_.end(); ++iter) {
      if (future == *iter) {
        futures_.erase(iter);
        break;
      }
    }
  }

  // Promote all elements from the last freezed position to the head.
  void Promote() {
    int nr_cnt = futures_.size() - last_freezed_size_;
    assert(nr_cnt > 0);
    int nr_notifying = 0;
    for (const auto &future : futures_) {
      if (future->IsError() || future->IsReady()) {
        nr_notifying += 1;
      } else {
        break;
      }
    }
    auto future = futures_.back();
    futures_.pop_back();
    futures_.insert(std::next(futures_.cbegin(), nr_notifying + 1), future);
    while (--nr_cnt) {
      auto future = futures_.back();
      futures_.pop_back();
      futures_.insert(std::next(futures_.cbegin(), nr_notifying), future);
    }
  }

  void Notify() { ev_async_send(loop_, &async_); }

private:
  static void OnNotify(struct ev_loop *loop, struct ev_async *w, int revent) {
    auto notifier = reinterpret_cast<AsyncNotifier *>(w->data);
    bool again = true;
    while (again) {
      auto future = notifier->futures_.front();
      if (!future || (!future->IsReady() && !future->IsError())) {
        break;
      }
      notifier->futures_.pop_front();
      if (notifier->futures_.empty()) {
        again = false;
      }
      // NB: Futures may get destruction in the following call.
      // As a result, this object maybe be destroyed cascadely.
      future->GetNotified();
    }
  }

private:
  struct ev_loop *loop_;
  struct ev_async async_;
  // Futures sorted by latent completion time.
  // NB: Pointers are borrowed here.
  std::list<Futurable *> futures_;
  int last_freezed_size_;
};

template <typename T, template<typename> class FutureType = Future>
std::shared_ptr<FutureType<T>>
MakeReadyFuture(std::shared_ptr<AsyncNotifier> notifier, T &v);
template <typename T, template<typename> class FutureType = Future>
std::shared_ptr<FutureType<T>>
MakeReadyFuture(std::shared_ptr<AsyncNotifier> notifier, T &&v);
template<template<typename> class FutureType = Future>
std::shared_ptr<FutureType<void>>
MakeReadyFuture(std::shared_ptr<AsyncNotifier> notifier);

template <typename T, typename = void> struct Futurize {
  using type = Future<T>;
  template <typename F,
            typename = typename std::enable_if_t<std::is_same<F, T>::value>>
  static std::shared_ptr<type> Apply(std::shared_ptr<AsyncNotifier> notifier,
                                     F &&v) {
    return MakeReadyFuture(notifier, std::forward<F>(v));
  }
};

template <typename T> struct Futurize<std::shared_ptr<Future<T>>> {
  using type = Future<T>;
  static std::shared_ptr<Future<T>>
  Apply(std::shared_ptr<AsyncNotifier> notifier, std::shared_ptr<Future<T>> v) {
    return v;
  }
};

template <typename... Futures>
struct Futurize<std::shared_ptr<WhenAll<std::shared_ptr<Futures>...>>> {
  using type = Future<typename WhenAll<std::shared_ptr<Futures>...>::Output>;
  static std::shared_ptr<type>
  Apply(std::shared_ptr<AsyncNotifier> notifier,
        std::shared_ptr<WhenAll<std::shared_ptr<Futures>...>> v) {
    return v;
  }
};

template<typename Fn, typename Rep, typename Period>
auto Delay(std::shared_ptr<FutureContext> ctx, 
        const std::chrono::duration<Rep, Period>& duration, Fn&& operation) -> std::shared_ptr<typename Futurize<decltype(std::declval<Fn>()())>::type>;

template <typename T> class Future : public Futurable {
public:
  using Output = T;

  using ThenFnArg = typename std::conditional_t<
      std::is_fundamental<T>::value, T,
      std::add_lvalue_reference_t<std::add_const_t<T>>>;
  template <typename Fn>
  using ThenHandleRet = decltype(
      std::declval<Fn>()(std::declval<std::shared_ptr<AsyncNotifier>>(),
                         std::declval<ThenFnArg>()));
  template <typename Fn>
  using ThenFnRet = typename Futurize<ThenHandleRet<Fn>>::type;

  PIN_TYPE(Future);

  optional<T> Get(bool block = true) const {
    while (block && !IsReady()) {}
    return FutureState::Pending != state_.load() ? value_ : optional<T>();
  }

  bool IsShareTheSameContext(std::shared_ptr<FutureContext> ctx) const {
    return ctx == notifier_;
  }

  bool IsReady() const override final {
    return FutureState::Pending != state_.load();
  }

  bool IsError() const override final {
    return FutureState::Error == state_.load();
  }

  // NB: Call this method on the same future multiple times is undefined
  // behavior.
  template <typename Fn>
  auto Then(Fn &&handle) -> std::shared_ptr<ThenFnRet<Fn>> {
    return _Then<Fn>(std::forward<Fn>(handle), std::integral_constant < bool,
                     std::is_void<ThenHandleRet<Fn>>::value ||
                         std::is_same<ThenHandleRet<Fn>, Future<void>>::value >
                             {});
  }

  // NB: This overload handles gcc's bug which is that gcc can't handle lambda
  // default parameters
  template <typename Fn, typename ErrFn>
  auto Then(Fn &&handle, ErrFn &&err_handle) -> std::shared_ptr<ThenFnRet<Fn>> {
    return _Then<Fn>(std::forward<Fn>(handle), std::forward<ErrFn>(err_handle),
                     std::integral_constant < bool,
                     std::is_void<ThenHandleRet<Fn>>::value ||
                         std::is_same<ThenHandleRet<Fn>, Future<void>>::value >
                             {});
  }

  // TODO: These methods should be moved to protected scope
  Future(std::shared_ptr<AsyncNotifier> notifier)
      : state_(FutureState::Pending), notifier_(notifier) {
    notifier_->Chained(this);
  }

  void Ref(std::shared_ptr<Future<T>> self) {
    assert(self.get() == this);
    self_ = self;
  }

  bool SetValue(const T &v) {
    if (FutureState::Pending == state_.load()) {
      assert(!value_);
      value_.emplace(v);
      state_.store(FutureState::Done);
      notifier_->Notify();
      return true;
    }
    return false;
  }

  bool SetValue(T &&v) {
    if (FutureState::Pending == state_.load()) {
      assert(!value_);
      value_.emplace(std::move(v));
      state_.store(FutureState::Done);
      notifier_->Notify();
      return true;
    }
    return false;
  }

protected:
  friend class Promise<T>;
  friend class Promise<T, SharedFuture>;

  template <typename O, template<typename> class FutureType>
  std::shared_ptr<FutureType<O>> friend MakeReadyFuture(
      std::shared_ptr<AsyncNotifier> notifier, O &v);
  template <typename O, template<typename> class FutureType>
  std::shared_ptr<FutureType<O>> friend MakeReadyFuture(
      std::shared_ptr<AsyncNotifier> notifier, O &&v);
  template <template<typename> class FutureType>
  std::shared_ptr<FutureType<void>> friend MakeReadyFuture(
      std::shared_ptr<AsyncNotifier> notifier);

  void GetNotified() override {
    assert(FutureState::Pending != state_.load());
    if (then_cb_) {
      then_cb_(notifier_, state_.load(), std::move(value_));
    }
    Unref();
  }

  void Unref() { self_.reset(); }

  template <typename Fn>
  auto _Then(Fn &&handle, std::true_type /* is void */)
      -> std::shared_ptr<Future<void>>;

  template <typename Fn>
  auto _Then(Fn &&handle, std::false_type /* isn't void */)
      -> std::shared_ptr<ThenFnRet<Fn>>;

  template <typename Fn, typename ErrFn>
  auto _Then(Fn &&handle, ErrFn &&err_handle, std::true_type /* is void */)
      -> std::shared_ptr<Future<void>>;

  template <typename Fn, typename ErrFn>
  auto _Then(Fn &&handle, ErrFn &&err_handle, std::false_type /* isn't void */)
      -> std::shared_ptr<ThenFnRet<Fn>>;

  bool SetError() {
    if (FutureState::Pending == state_.load()) {
      state_.store(FutureState::Error);
      notifier_->Notify();
      return true;
    }
    return false;
  }

protected:
  std::atomic<FutureState> state_;
  optional<T> value_;
  std::function<void(std::shared_ptr<AsyncNotifier>, FutureState,
                     optional<T> &&)>
      then_cb_;
  std::shared_ptr<AsyncNotifier> notifier_;
  std::shared_ptr<Future<T>> self_;
};

template <> class Future<void> : public Futurable {
public:
  using Output = void;
  template <typename Fn>
  using ThenHandleRet = decltype(
      std::declval<Fn>()(std::declval<std::shared_ptr<AsyncNotifier>>()));
  template <typename Fn>
  using ThenFnRet = typename Futurize<ThenHandleRet<Fn>>::type;

  PIN_TYPE(Future);

  // NB: Calling Future<void>::Get(false) equals to a nop.
  void Get(bool block = true) const {
    while (block && !IsReady()) {}
  }

  bool IsShareTheSameContext(std::shared_ptr<FutureContext> ctx) const {
    return ctx == notifier_;
  }

  bool IsReady() const override final {
    return FutureState::Pending != state_.load();
  }

  bool IsError() const override final {
    return FutureState::Error == state_.load();
  }

  // NB: Call this method on the same future multiple times chains
  // handles by invocation order.
  template <typename Fn>
  auto Then(Fn &&handle) -> std::shared_ptr<ThenFnRet<Fn>> {
    return _Then<Fn>(std::forward<Fn>(handle), std::integral_constant < bool,
                     std::is_void<ThenHandleRet<Fn>>::value ||
                         std::is_same<ThenHandleRet<Fn>, Future<void>>::value >
                             {});
  }

  template <typename Fn, typename ErrFn>
  auto Then(Fn &&handle, ErrFn &&err_handle) -> std::shared_ptr<ThenFnRet<Fn>> {
    return _Then<Fn>(std::forward<Fn>(handle), std::forward<ErrFn>(err_handle),
                     std::integral_constant < bool,
                     std::is_void<ThenHandleRet<Fn>>::value ||
                         std::is_same<ThenHandleRet<Fn>, Future<void>>::value >
                             {});
  }

  // TODO: These methods should be moved to protected scope
  Future(std::shared_ptr<AsyncNotifier> notifier)
      : state_(FutureState::Pending), notifier_(notifier) {
    notifier_->Chained(this);
  }

  void Ref(std::shared_ptr<Future<void>> self) {
    assert(self.get() == this);
    self_ = self;
  }

  bool SetValue() {
    if (FutureState::Pending == state_.load()) {
      state_.store(FutureState::Done);
      notifier_->Notify();
      return true;
    }
    return false;
  }

protected:
  friend class Promise<void>;
  friend class Promise<void, SharedFuture>;

  template <typename O, template<typename> class FutureType>
  std::shared_ptr<FutureType<O>> friend MakeReadyFuture(
      std::shared_ptr<AsyncNotifier> notifier, O &v);
  template <typename O, template<typename> class FutureType>
  std::shared_ptr<FutureType<O>> friend MakeReadyFuture(
      std::shared_ptr<AsyncNotifier> notifier, O &&v);
  template <template<typename> class FutureType>
  std::shared_ptr<FutureType<void>> friend MakeReadyFuture(
      std::shared_ptr<AsyncNotifier> notifier);

  template <typename Fn>
  auto _Then(Fn &&handle, std::true_type /* is void */)
      -> std::shared_ptr<Future<void>>;

  template <typename Fn>
  auto _Then(Fn &&handle, std::false_type /* isn't void */)
      -> std::shared_ptr<ThenFnRet<Fn>>;

  template <typename Fn, typename ErrFn>
  auto _Then(Fn &&handle, ErrFn &&err_handle, std::true_type /* is void */)
      -> std::shared_ptr<Future<void>>;

  template <typename Fn, typename ErrFn>
  auto _Then(Fn &&handle, ErrFn &&err_handle, std::false_type /* isn't void */)
      -> std::shared_ptr<ThenFnRet<Fn>>;

  void Unref() { self_.reset(); }

  void GetNotified() override {
    assert(FutureState::Pending != state_.load());
    if (then_cb_) {
      then_cb_(notifier_, state_.load());
    }
    Unref();
  }

  bool SetError() {
    if (FutureState::Pending == state_.load()) {
      state_.store(FutureState::Error);
      notifier_->Notify();
      return true;
    }
    return false;
  }

protected:
  std::atomic<FutureState> state_;
  std::function<void(std::shared_ptr<AsyncNotifier>, FutureState)> then_cb_;
  std::shared_ptr<AsyncNotifier> notifier_;
  std::shared_ptr<Future<void>> self_;
};

// This kind of futures allow for multiple futures wait on it
template <typename T> class SharedFuture final : public Future<T> {
public:
  SharedFuture(std::shared_ptr<AsyncNotifier> notifier)
      : Future<T>(notifier) {}

  template <typename Fn>
  auto Then(std::shared_ptr<AsyncNotifier> notifier, Fn &&handle)
      -> std::shared_ptr<typename Future<T>::template ThenFnRet<Fn>>;

protected:
  void GetNotified() override {
    assert(this->IsReady());
    if (!this->IsError()) {
      std::for_each(subscribers_.begin(), subscribers_.end(),
                    [this](auto &s) { s.SetValue(this->Get(false).value()); });
    } else {
      std::for_each(subscribers_.begin(), subscribers_.end(),
                    [](auto &s) { s.SetError(); });
    }
    this->Unref();
  }

private:
  std::vector<Promise<T>> subscribers_;
};

template <> class SharedFuture<void> final : public Future<void> {
public:
  SharedFuture(std::shared_ptr<AsyncNotifier> notifier)
      : Future<void>(notifier) {}

  template <typename Fn>
  auto Then(std::shared_ptr<AsyncNotifier> notifier, Fn &&handle)
      -> std::shared_ptr<typename Future<void>::template ThenFnRet<Fn>>;

protected:
  void GetNotified() override {
    assert(this->IsReady());
    if (!this->IsError()) {
      std::for_each(subscribers_.begin(), subscribers_.end(),
                    [](auto &s) { s.SetValue(); });
    } else {
      std::for_each(subscribers_.begin(), subscribers_.end(),
                    [](auto &s) { s.SetError(); });
    }
    this->Unref();
  }

private:
  std::vector<Promise<void>> subscribers_;
};

template <typename T, template<typename> class FutureT> class Promise {
    static_assert(std::is_base_of<Futurable, FutureT<T>>::value, "`FutureT` should be a derived class of `Futurable`");
public:
  Promise(std::shared_ptr<AsyncNotifier> notifier) noexcept
      : value_(std::shared_ptr<FutureT<T>>(new FutureT<T>(notifier))) {
    value_->Ref(value_);
  }
  Promise(Promise &&o) = default;
  std::shared_ptr<FutureT<T>> GetFuture() { return value_; }
  void SetValue(const T &value) { value_->SetValue(value); }
  void SetValue(T &&value) { value_->SetValue(std::move(value)); }
  void SetError() { value_->SetError(); }

private:
  std::shared_ptr<FutureT<T>> value_;
};

template <template<typename> class FutureT> class Promise<void, FutureT> {
    static_assert(std::is_base_of<Futurable, FutureT<void>>::value, "`FutureT` should be a derived class of `Futurable`");
public:
  Promise(std::shared_ptr<AsyncNotifier> notifier) noexcept
      : value_(std::shared_ptr<FutureT<void>>(new FutureT<void>(notifier))) {
    value_->Ref(value_);
  }
  Promise(Promise &&o) = default;
  std::shared_ptr<FutureT<void>> GetFuture() { return value_; }
  void SetValue() { value_->SetValue(); }
  void SetError() { value_->SetError(); }

private:
  std::shared_ptr<FutureT<void>> value_;
};

template<typename T>
struct Pipe {
    void operator()(std::shared_ptr<Future<T>> future, std::shared_ptr<Promise<T>> promise) {
        future->Then([promise](
                    auto &&ctx,
                    const auto &v) noexcept { promise->SetValue(v); },
                [promise](auto &&ctx) noexcept { promise->SetError(); });
    }
};

template<>
struct Pipe<void>{ 
    void operator()(std::shared_ptr<Future<void>> future, std::shared_ptr<Promise<void>> promise) {
        future->Then([promise](auto &&ctx) noexcept { promise->SetValue(); },
                            [promise](auto &&ctx) noexcept { promise->SetError(); });
    }
};

template<typename Fn, typename Rep, typename Period>
auto Delay(std::shared_ptr<FutureContext> notifier, 
        const std::chrono::duration<Rep, Period>& duration, Fn&& operation) -> std::shared_ptr<typename Futurize<decltype(std::declval<Fn>()())>::type> {
    using namespace std::chrono;

    auto promise = std::make_shared<Promise<typename Futurize<decltype(std::declval<Fn>()())>::type::Output>>(notifier);
    auto ret = promise->GetFuture();
    auto timer = static_cast<struct ev_timer*>(malloc(sizeof(struct ev_timer)));
    auto cb = [](struct ev_loop *loop, struct ev_timer *w, int revent) {
        assert(w->data != nullptr);
        auto arg = reinterpret_cast<std::tuple<decltype(notifier), decltype(promise), Fn>*>(w->data);
        try {
            // The same code as `Future::_Then`...
            std::get<0>(*arg)->FreezeOrder();
            auto future = Futurize<decltype(std::declval<Fn>()())>::Apply(std::get<0>(*arg), std::get<2>(*arg)());
            Pipe<typename decltype(future)::element_type::Output>{}(future, std::get<1>(*arg));
            if (future->IsShareTheSameContext(std::get<0>(*arg))) {
                std::get<0>(*arg)->Promote();
            }
        } catch (const std::exception &e) {
            std::get<1>(*arg)->SetError();
        }
        delete arg;
        w->data = nullptr;
    };
    ev_timer_init(timer, cb, duration_cast<seconds>(duration).count(), 0);
    timer->data = new std::tuple<decltype(notifier), decltype(promise), Fn>(notifier, std::move(promise), std::forward<Fn>(operation));
    notifier->RunTimer(timer);
    return ret; 
}

template <typename T>
template <typename Fn>
auto Future<T>::_Then(Fn &&handle, std::true_type /* is void */)
    -> std::shared_ptr<Future<void>> {
  auto promise = std::make_shared<Promise<void>>(notifier_);
  auto ret = promise->GetFuture();
  then_cb_ =
      [ handle = std::forward<Fn>(handle), promise = std::move(promise) ](
          std::shared_ptr<AsyncNotifier> notifier, FutureState fs,
          optional<T> && opt) noexcept {
    if (FutureState::Done == fs) {
      try {
        handle(notifier, opt.value());
        promise->SetValue();
      } catch (const std::exception &e) {
        promise->SetError();
      }
    } else {
      promise->SetError();
    }
  };

  return ret;
}

template <typename T>
template <typename Fn>
auto Future<T>::_Then(Fn &&handle, std::false_type /* isn't void */)
    -> std::shared_ptr<ThenFnRet<Fn>> {
  auto promise =
      std::make_shared<Promise<typename ThenFnRet<Fn>::Output>>(notifier_);
  auto ret = promise->GetFuture();
  then_cb_ =
      [ handle = std::forward<Fn>(handle), promise = std::move(promise) ](
          std::shared_ptr<AsyncNotifier> notifier, FutureState fs,
          optional<T> && opt) noexcept {
    if (FutureState::Done == fs) {
      try {
        notifier->FreezeOrder();
        auto future = Futurize<ThenHandleRet<Fn>>::Apply(
            notifier, handle(notifier, opt.value()));
        Pipe<typename decltype(future)::element_type::Output>{}(future, promise);
        if (future->IsShareTheSameContext(notifier)) {
          // NB: we have chained some futures onto the notifier. Thus,
          // in consideration of other futures that already exist on the
          // notifier, we need to adjust the order of them to make sure they are
          // notifed before others.
          notifier->Promote();
        }
      } catch (const std::exception &e) {
        promise->SetError();
      }
    } else {
      promise->SetError();
    }
  };

  return ret;
}

template <typename T>
template <typename Fn, typename ErrFn>
auto Future<T>::_Then(Fn &&handle, ErrFn &&err_handle,
                      std::true_type /* is void */)
    -> std::shared_ptr<Future<void>> {
  auto promise = std::make_shared<Promise<void>>(notifier_);
  auto ret = promise->GetFuture();
  then_cb_ = [handle = std::forward<Fn>(handle),
              err_handle = std::forward<ErrFn>(err_handle),
              promise =
                  std::move(promise)](std::shared_ptr<AsyncNotifier> notifier,
                                      FutureState fs, optional<T> &&opt) {
    if (FutureState::Done == fs) {
      try {
        handle(notifier, opt.value());
        promise->SetValue();
      } catch (const std::exception &e) {
        promise->SetError();
      }
    } else {
      err_handle(notifier);
      promise->SetError();
    }
  };

  return ret;
}

template <typename T>
template <typename Fn, typename ErrFn>
auto Future<T>::_Then(Fn &&handle, ErrFn &&err_handle,
                      std::false_type /* isn't void */)
    -> std::shared_ptr<ThenFnRet<Fn>> {
  auto promise =
      std::make_shared<Promise<typename ThenFnRet<Fn>::Output>>(notifier_);
  auto ret = promise->GetFuture();
  then_cb_ = [handle = std::forward<Fn>(handle),
              err_handle = std::forward<ErrFn>(err_handle),
              promise =
                  std::move(promise)](std::shared_ptr<AsyncNotifier> notifier,
                                      FutureState fs, optional<T> &&opt) {
    if (FutureState::Done == fs) {
      try {
        notifier->FreezeOrder();
        auto future = Futurize<ThenHandleRet<Fn>>::Apply(
            notifier, handle(notifier, opt.value()));
        Pipe<typename decltype(future)::element_type::Output>{}(future, promise);
        if (future->IsShareTheSameContext(notifier)) {
          notifier->Promote();
        }
      } catch (const std::exception &e) {
        promise->SetError();
      }
    } else {
      err_handle(notifier);
      promise->SetError();
    }
  };

  return ret;
}

template <typename Fn>
auto Future<void>::_Then(Fn &&handle, std::true_type /* is void */)
    -> std::shared_ptr<Future<void>> {
  auto promise = std::make_shared<Promise<void>>(notifier_);
  auto ret = promise->GetFuture();
  then_cb_ =
      [ handle = std::forward<Fn>(handle), promise = std::move(promise) ](
          std::shared_ptr<AsyncNotifier> notifier, FutureState fs) noexcept {
    if (FutureState::Done == fs) {
      try {
        handle(notifier);
        promise->SetValue();
      } catch (const std::exception &e) {
        promise->SetError();
      }
    } else {
      promise->SetError();
    }
  };

  return ret;
}

template <typename Fn>
auto Future<void>::_Then(Fn &&handle, std::false_type /* isn't void */)
    -> std::shared_ptr<ThenFnRet<Fn>> {
  auto promise =
      std::make_shared<Promise<typename ThenFnRet<Fn>::Output>>(notifier_);
  auto ret = promise->GetFuture();
  then_cb_ =
      [ handle = std::forward<Fn>(handle), promise = std::move(promise) ](
          std::shared_ptr<AsyncNotifier> notifier, FutureState fs) noexcept {
    if (FutureState::Done == fs) {
      try {
        notifier->FreezeOrder();
        auto future =
            Futurize<ThenHandleRet<Fn>>::Apply(notifier, handle(notifier));
        Pipe<typename decltype(future)::element_type::Output>{}(future, promise);
        if (future->IsShareTheSameContext(notifier)) {
          notifier->Promote();
        }
      } catch (const std::exception &e) {
        promise->SetError();
      }
    } else {
      promise->SetError();
    }
  };

  return ret;
}

template <typename Fn, typename ErrFn>
auto Future<void>::_Then(Fn &&handle, ErrFn &&err_handle,
                         std::true_type /* is void */)
    -> std::shared_ptr<Future<void>> {
  auto promise = std::make_shared<Promise<void>>(notifier_);
  auto ret = promise->GetFuture();
  then_cb_ = [handle = std::forward<Fn>(handle),
              err_handle = std::forward<ErrFn>(err_handle),
              promise = std::move(promise)](
                 std::shared_ptr<AsyncNotifier> notifier, FutureState fs) {
    if (FutureState::Done == fs) {
      try {
        handle(notifier);
        promise->SetValue();
      } catch (const std::exception &e) {
        promise->SetError();
      }
    } else {
      err_handle(notifier);
      promise->SetError();
    }
  };

  return ret;
}

template <typename Fn, typename ErrFn>
auto Future<void>::_Then(Fn &&handle, ErrFn &&err_handle,
                         std::false_type /* isn't void */)
    -> std::shared_ptr<ThenFnRet<Fn>> {
  auto promise =
      std::make_shared<Promise<typename ThenFnRet<Fn>::Output>>(notifier_);
  auto ret = promise->GetFuture();
  then_cb_ = [handle = std::forward<Fn>(handle),
              err_handle = std::forward<ErrFn>(err_handle),
              promise = std::move(promise)](
                 std::shared_ptr<AsyncNotifier> notifier, FutureState fs) {
    if (FutureState::Done == fs) {
      try {
        notifier->FreezeOrder();
        auto future =
            Futurize<ThenHandleRet<Fn>>::Apply(notifier, handle(notifier));
        Pipe<typename decltype(future)::element_type::Output>{}(future, promise);
        if (future->IsShareTheSameContext(notifier)) {
          notifier->Promote();
        }
      } catch (const std::exception &e) {
        promise->SetError();
      }
    } else {
      err_handle(notifier);
      promise->SetError();
    }
  };

  return ret;
}

template <typename T>
template <typename Fn>
auto SharedFuture<T>::Then(std::shared_ptr<AsyncNotifier> notifier, Fn &&handle)
    -> std::shared_ptr<typename Future<T>::template ThenFnRet<Fn>> {
  if (!this->IsReady()) {
    subscribers_.emplace_back(notifier);
    auto ret = subscribers_.back().GetFuture();
    return ret->Then(std::forward<Fn>(handle));
  } else {
    Promise<T> p(notifier);
    if (!this->IsError()) {
        p.SetValue(this->Get(false).value());
    } else {
        p.SetError();
    }
    return p.GetFuture()->Then(std::forward<Fn>(handle));
  }
}

template <typename Fn>
auto SharedFuture<void>::Then(std::shared_ptr<AsyncNotifier> notifier,
                              Fn &&handle)
    -> std::shared_ptr<typename Future<void>::template ThenFnRet<Fn>> {
  if (!this->IsReady()) {
    subscribers_.emplace_back(notifier);
    auto ret = subscribers_.back().GetFuture();
    return ret->Then(std::forward<Fn>(handle));
  } else {
    Promise<void> p(notifier);
    if (!this->IsError()) {
        p.SetValue();
    } else {
        p.SetError();
    }
    return p.GetFuture()->Then(std::forward<Fn>(handle));
  }
}

// Make a Future<T> that's immediately ready
template <typename T, template<typename> class FutureType>
std::shared_ptr<FutureType<T>>
MakeReadyFuture(std::shared_ptr<AsyncNotifier> notifier, T &v) {
  auto ret = std::shared_ptr<FutureType<T>>(new FutureType<T>(notifier));
  ret->Ref(ret);
  ret->SetValue(std::forward<T>(v));
  return ret;
}

template <typename T, template<typename> class FutureType>
std::shared_ptr<FutureType<T>>
MakeReadyFuture(std::shared_ptr<AsyncNotifier> notifier, T &&v) {
  auto ret = std::shared_ptr<FutureType<T>>(new FutureType<T>(notifier));
  ret->Ref(ret);
  ret->SetValue(std::forward<T>(v));
  return ret;
}

template <template<typename> class FutureType>
inline std::shared_ptr<FutureType<void>>
MakeReadyFuture(std::shared_ptr<AsyncNotifier> notifier) {
  auto ret = std::shared_ptr<FutureType<void>>(new FutureType<void>(notifier));
  ret->Ref(ret);
  ret->SetValue();
  return ret;
}

template <typename... Futures> struct FutureCombinator;

template <template <typename T> class Future, typename T>
struct FutureCombinator<std::shared_ptr<Future<T>>> {
  using Output = std::tuple<optional<T>>;
  using Parent = std::tuple<std::shared_ptr<Future<T>>>;
};

template <template <typename T> class Future, typename T, typename... Futures>
struct FutureCombinator<std::shared_ptr<Future<T>>, Futures...> {
  using Output = decltype(std::tuple_cat(
      std::declval<std::tuple<optional<T>>>(),
      std::declval<typename FutureCombinator<Futures...>::Output>()));
  using Parent = decltype(std::tuple_cat(
      std::declval<std::tuple<std::shared_ptr<Future<T>>>>(),
      std::declval<typename FutureCombinator<Futures...>::Parent>()));
};

template <std::size_t Index, typename Handle>
void FutureIterate(std::add_pointer_t<Handle>) {}

template <std::size_t Index, typename Handle, typename Future,
          typename... LeftFutures>
void FutureIterate(std::add_pointer_t<Handle> handle, Future &&future,
                   LeftFutures &&... lefts) {
  handle->template Apply<Index>(std::forward<Future>(future));
  FutureIterate<Index + 1, Handle, decltype(lefts)...>(
      handle, std::forward<LeftFutures>(lefts)...);
}

template <typename... Futures> struct IsFutureCopyConstructible {
  static constexpr bool value = false;
};

template <template <typename T> class Future, typename T>
struct IsFutureCopyConstructible<std::shared_ptr<Future<T>>> {
  static constexpr bool value = std::is_copy_constructible<T>::value;
};

template <template <typename T> class Future, typename T, typename... Futures>
struct IsFutureCopyConstructible<std::shared_ptr<Future<T>>, Futures...> {
  static constexpr bool value = std::is_copy_constructible<T>::value &&
                                IsFutureCopyConstructible<Futures...>::value;
};

template <typename... Futures>
class WhenAll final
    : public Future<typename FutureCombinator<Futures...>::Output> {
public:
  using Output = typename FutureCombinator<Futures...>::Output;
  using Parent = typename FutureCombinator<Futures...>::Parent;

  WhenAll(std::shared_ptr<FutureContext> ctx, Futures... futures)
      : Future<Output>(ctx), nr_waitings_(sizeof...(futures)) {
    static_assert(
        IsFutureCopyConstructible<Futures...>::value,
        "Each T in Future<T> in when_all should be copy constructible.");
    this->value_.emplace();
    FutureIterate<0, WhenAll, Futures...>(this,
                                          std::forward<Futures>(futures)...);
  }
  PIN_TYPE(WhenAll);

  optional<Output> Get(bool block) = delete;
  bool SetValue(Output &) = delete;
  bool SetValue(Output &&) = delete;
  bool SetError() = delete;

private:
  template <std::size_t Index, typename Handle, typename Future,
            typename... LeftFutures>
  friend void FutureIterate(std::add_pointer_t<Handle> handle, Future &&future,
                            LeftFutures &&... lefts);
  friend std::shared_ptr<WhenAll<Futures...>>
  MakeWhenAll<Futures...>(std::shared_ptr<FutureContext> ctx,
                          Futures &&... futures);

  template <std::size_t Index, typename Future> void Apply(Future &&future) {
    future->Then(
        [this](auto &&ctx, auto &&v) {
          std::get<Index>(this->value_.value()).emplace(v);
          if (1 >= this->nr_waitings_.fetch_sub(1)) {
            this->state_.store(FutureState::Done);
            this->notifier_->Notify();
          }
        },
        [this](auto &&ctx) {
          if (1 >= this->nr_waitings_.fetch_sub(1)) {
            this->state_.store(FutureState::Done);
            this->notifier_->Notify();
          }
        });
  }

private:
  std::atomic_int nr_waitings_;
};

template <typename T>
class WhenEach final : public Future<std::vector<optional<T>>> {
public:
  using Output = std::vector<optional<T>>;

  WhenEach(std::shared_ptr<FutureContext> ctx)
      : Future<Output>(ctx), nr_waitings_(0), freezed_(false) {
    this->value_.emplace();
  }
  PIN_TYPE(WhenEach);

  optional<Output> Get(bool block) = delete;
  bool SetValue(Output &) = delete;
  bool SetValue(Output &&) = delete;
  bool SetError() = delete;

  void WaitOn(std::shared_ptr<Future<T>> future) {
    const auto position = nr_waitings_.fetch_add(1);
    this->value_->emplace_back();
    this->state_.store(FutureState::Pending);
    future->Then(
        [this, position](auto &&ctx, auto &&v) {
          this->value_->at(position) = v;
          if (1 >= nr_waitings_.fetch_sub(1) && freezed_.load()) {
            this->state_.store(FutureState::Done);
            this->notifier_->Notify();
          }
        },
        [this](auto &&ctx) {
          if (1 >= nr_waitings_.fetch_sub(1) && freezed_.load()) {
            this->state_.store(FutureState::Done);
            this->notifier_->Notify();
          }
        });
  }

  // The future may destroy itself when nr_waitings_ gets zero
  // after being freezed.
  // Conversely, the future may never be destroyed if it's not
  // freezed.
  void Freeze() {
    freezed_.store(true);
    if (0 >= nr_waitings_.load()) {
      this->state_.store(FutureState::Done);
      this->notifier_->Notify();
    }
  }

private:
  friend std::shared_ptr<WhenEach<T>>
  MakeWhenEach<T>(std::shared_ptr<FutureContext> ctx);

  void GetNotified() override final {
    assert(FutureState::Pending != this->state_.load());
    assert(freezed_.load() == true);
    if (this->then_cb_) {
      this->then_cb_(this->notifier_, this->state_.load(),
                     std::move(this->value_));
    }
    this->Unref();
  }

private:
  std::atomic_int nr_waitings_;
  std::atomic_bool freezed_;
};

template <> class WhenEach<void> final : public Future<void> {
public:
  using Output = void;

  WhenEach(std::shared_ptr<FutureContext> ctx)
      : Future<Output>(ctx), nr_waitings_(0), freezed_(false) {}
  PIN_TYPE(WhenEach);

  optional<Output> Get(bool block) = delete;
  bool SetValue() = delete;
  bool SetError() = delete;

  void WaitOn(std::shared_ptr<Future<void>> future) {
    nr_waitings_.fetch_add(1);
    this->state_.store(FutureState::Pending);
    future->Then(
        [this](auto &&ctx) {
          if (1 >= nr_waitings_.fetch_sub(1) && freezed_.load()) {
            this->state_.store(FutureState::Done);
            this->notifier_->Notify();
          }
        },
        [this](auto &&ctx) {
          if (1 >= nr_waitings_.fetch_sub(1) && freezed_.load()) {
            this->state_.store(FutureState::Done);
            this->notifier_->Notify();
          }
        });
  }

  // The future may destroy itself when nr_waitings_ gets zero
  // after being freezed.
  // Conversely, the future may never be destroyed if it's not
  // freezed.
  void Freeze() {
    freezed_.store(true);
    if (0 >= nr_waitings_.load()) {
      this->state_.store(FutureState::Done);
      this->notifier_->Notify();
    }
  }

private:
  friend std::shared_ptr<WhenEach<void>>
  MakeWhenEach<void>(std::shared_ptr<FutureContext> ctx);

  void GetNotified() override final {
    assert(FutureState::Pending != this->state_.load());
    assert(freezed_.load());
    if (this->then_cb_) {
      this->then_cb_(this->notifier_, this->state_.load());
    }
  }

private:
  std::atomic_int nr_waitings_;
  std::atomic_bool freezed_;
};

template <typename... Futures>
std::shared_ptr<WhenAll<Futures...>>
MakeWhenAll(std::shared_ptr<FutureContext> ctx, Futures &&... futures) {
  auto ret = std::make_shared<WhenAll<Futures...>>(
      ctx, std::forward<Futures>(futures)...);
  ret->Ref(ret);
  return ret;
}

template <typename T>
std::shared_ptr<WhenEach<T>> MakeWhenEach(std::shared_ptr<FutureContext> ctx) {
  auto ret = std::make_shared<WhenEach<T>>(ctx);
  ret->Ref(ret);
  return ret;
}

#undef DISABLE_COPY
#undef DISABLE_MOVE
#undef PIN_TYPE

} // namespace cps
