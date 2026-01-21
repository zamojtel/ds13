use bitvec::vec::BitVec;
use module_system::{Handler, ModuleRef};
use std::cmp::min;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use uuid::Uuid;

/// A source of randomness.
pub(crate) trait RandomnessSource {
    /// Generates a next pseudo-random u32 value selected
    /// from a uniform distribution.
    fn next_u32(&mut self) -> u32;
}

/// A conflict-free state-based replicated counter.
///
/// `T` is the final result type we are interested in. Among others, it defines "infinity".
/// The trait is implemented on the "backend type" holding the intermediary values.
pub(crate) trait ConflictFreeReplicatedCounter<T> {
    /// Sets a given counter so that it counts no elements.
    fn set_to_zero(&mut self);

    /// Sets a given counter so that it counts
    /// an infinite number of elements (all possible).
    fn set_to_infinity(&mut self);

    /// Adds one more element to a given counter.
    ///
    /// That is, "increments the counter by one" using
    /// a given source of randomness.
    /// If the counter counts an infinite number of elements,
    /// an `Err` is returned and the given counter remains
    /// intact; otherwise, `Ok` is returned.
    fn try_count_one_more_element(&mut self, rs: &mut dyn RandomnessSource) -> Result<(), String>;

    /// Merges another counter with a given counter.
    ///
    /// As a result, the given counter counts
    /// elements counted originally by both itself
    /// and the other counter. If the two counters are
    /// incompatible, `Err` is returned and the given
    /// counter remains intact; otherwise, `Ok` is returned.
    fn try_merge_with(&mut self, other: &Self) -> Result<(), String>;

    /// Returns the number of elements counted by a given counter.
    ///
    /// You may use floating point operations in the implementation.
    /// If the value needs to be rounded, it should follow standard semantics
    /// (i.e., use the `round()` method).
    fn evaluate(&self) -> T;
}

/// An implementation of a probabilistic counting sketch.
#[derive(Clone, Debug)]
pub(crate) struct ProbabilisticCounter {
    // TODO: you may add any necessary fields here
    // For storing bit vectors, consider using `bitvec::vec::BitVec` (our Cargo dependency).
}

impl ProbabilisticCounter {
    /// The scaling factor used in probabilistic counting.
    const SCALING_FACTOR: f64 = 1.29281;

    /// Creates a new probabilistic counter.
    ///
    /// The counter has a given number of sketch instances and bits per instance.
    /// The allowed bits per instance values are: 8, 16, 24, and 32.
    /// The number of instances (and thus the total storage bits) is not explicitly bounded
    /// (but you may use naive summation and averaging in `evaluate`).
    /// The counter initially counts no elements.
    pub(crate) fn new_zero(bits_per_instance: usize, num_instances: usize) -> Self {
        assert!(num_instances > 0);
        assert!(bits_per_instance > 0 && bits_per_instance <= u32::BITS as usize);
        assert!(bits_per_instance.is_multiple_of(8));

        todo!()
    }

    /// Creates a new probabilistic counter with the same configuration as a given one.
    ///
    /// The new counter counts no elements.
    pub(crate) fn new_zero_with_same_config(other: &ProbabilisticCounter) -> Self {
        ProbabilisticCounter::new_zero(other.get_num_bits_per_instance(), other.get_num_instances())
    }

    /// Returns the number of sketch instances utilized
    /// by a given probabilistic counter.
    pub(crate) fn get_num_instances(&self) -> usize {
        todo!()
    }

    /// Returns the number of bits per sketch instance
    /// utilized by a given probabilistic counter.
    pub(crate) fn get_num_bits_per_instance(&self) -> usize {
        todo!()
    }

    /// Given a u32 bit number drawn at random from a
    /// uniform distribution produces a number from
    /// a geometric distribution with probability 1/2.
    ///
    /// The second parameter denotes the number of bins of the geometric distribution
    /// (i.e., the function returns a value from `[0, num_bits)` half-open range.)
    /// This function shall be used for selecting bits for
    /// incrementation of the sketches.
    pub(crate) fn uniform_u32_to_geometric(rand_no: u32, num_bits: usize) -> u32 {
        let trailing_zeroes = rand_no.trailing_zeros();
        // This implementation is slightly biased
        min(trailing_zeroes, (num_bits - 1) as u32)
    }

    // Methods only used for testing

    /// Returns a uniform random value that leads to
    /// setting a specific bit in the counter. In principle,
    /// this is used to partially revert function
    /// `uniform_u32_to_geometric` for testing.
    #[cfg(test)]
    pub(crate) fn geometric_to_sample_u32(bit_idx: u32) -> u32 {
        assert!(bit_idx < u32::BITS);
        1_u32 << bit_idx
    }

    /// Returns a given bit in a given instance of the sketch.
    ///
    /// The 0th bit is the one most likely to be set.
    #[cfg(test)]
    pub(crate) fn get_bit(&self, instance_idx: usize, in_instance_bit_idx: usize) -> bool {
        todo!()
    }

    /// Sets a given bit in a given instance of the sketch
    /// to the value provided as a parameter.
    ///
    /// The 0th bit is the one most likely to be set.
    /// As a result of this test method, the counter evaluation may change, and
    /// `get_bit` should return values accordingly.
    #[cfg(test)]
    pub(crate) fn set_bit(&mut self, instance_idx: usize, in_instance_bit_idx: usize, val: bool) {
        todo!()
    }
    // TODO: you may add any extra methods here
}

impl ConflictFreeReplicatedCounter<u64> for ProbabilisticCounter {
    fn set_to_zero(&mut self) {
        todo!()
    }

    fn set_to_infinity(&mut self) {
        todo!()
    }

    fn try_count_one_more_element(&mut self, rs: &mut dyn RandomnessSource) -> Result<(), String> {
        todo!()
    }

    fn try_merge_with(&mut self, other: &Self) -> Result<(), String> {
        todo!()
    }

    fn evaluate(&self) -> u64 {
        todo!()
    }
}

// -------------------------------------------
// Usage of the counter in a distributed system.
// -------------------------------------------

/// A service allowing for sampling random nodes
/// from the system for gossiping.
#[async_trait::async_trait]
pub(crate) trait PeerSamplingService {
    /// Returns a reference to a random Node
    /// in the system.
    async fn get_random_peer(&mut self) -> ModuleRef<Node>;
}

/// A node (process) in the system.
pub(crate) struct Node {
    uuid: Uuid,
    rs: Box<dyn RandomnessSource + Send>,
    pss: Box<dyn PeerSamplingService + Send>,
    // TODO: you may add any necessary fields here
}

/// A message used by a client to install a query on a node.
///
/// The query gets associated with the node's UUID.
/// The node stores only the last query, but previous queries for the node may
/// still be present in the overall system, and should eventually cease to be processed.
///
/// The `bits_per_instance` and `num_instances` are valid configurations of the
/// `ProbabilisticCounter`.
/// The query counts the number of nodes satisfying the `predicate`
/// (for this task sake, we just use their UUID).
pub(crate) struct QueryInstallMsg {
    pub(crate) bits_per_instance: usize,
    pub(crate) num_instances: usize,
    pub(crate) predicate: Arc<dyn Fn(&Uuid) -> bool + Send + Sync>,
}

/// A message used by a client to poll a node
/// to provide its current estimate of the query value.
pub(crate) struct QueryResultPollMsg {
    pub(crate) initiator: Uuid,
    pub(crate) callback: QueryResultPollCallback,
}

pub(crate) type QueryResultPollCallback =
    Box<dyn FnOnce(Option<u64>) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send>;

/// A message that triggers a node to initiate
/// gossiping.
pub(crate) struct SyncTriggerMsg {}

/// A gossip message sent between two nodes.
pub(crate) struct SyncGossipMsg {
    // TODO: you may add any necessary fields here
}

impl Node {
    pub(crate) fn new(
        uuid: Uuid,
        rs: Box<dyn RandomnessSource + Send>,
        pss: Box<dyn PeerSamplingService + Send>,
    ) -> Self {
        Self {
            uuid,
            rs,
            pss,
            // TODO: you may add initialization of any added fields here
        }
    }

    // TODO: you may add any extra methods here
}

#[async_trait::async_trait]
impl Handler<QueryInstallMsg> for Node {
    async fn handle(&mut self, msg: QueryInstallMsg) {
        if msg.bits_per_instance == 0
            || msg.bits_per_instance > u32::BITS as usize
            || !msg.bits_per_instance.is_multiple_of(8)
            || msg.num_instances == 0
        {
            return;
        }
        todo!()
    }
}

#[async_trait::async_trait]
impl Handler<QueryResultPollMsg> for Node {
    async fn handle(&mut self, msg: QueryResultPollMsg) {
        todo!()
    }
}

#[async_trait::async_trait]
impl Handler<SyncTriggerMsg> for Node {
    async fn handle(&mut self, _msg: SyncTriggerMsg) {
        todo!()
    }
}

#[async_trait::async_trait]
impl Handler<SyncGossipMsg> for Node {
    async fn handle(&mut self, msg: SyncGossipMsg) {
        todo!()
    }
}
