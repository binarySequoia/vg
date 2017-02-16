#include "variant_adder.hpp"
#include "mapper.hpp"

namespace vg {

using namespace std;

VariantAdder::VariantAdder(VG& graph) : graph(graph), sync(graph) {
    graph.paths.for_each_name([&](const string& name) {
        // Save the names of all the graph paths, so we don't need to lock the
        // graph to check them.
        path_names.insert(name);
    });
    
    // Show progress if the graph does.
    show_progress = graph.show_progress;
    
    // Make sure to dice nodes to 1024 or smaller, the max size that GCSA2
    // supports, in case we need to GCSA-index part of the graph.
    graph.dice_nodes(1024);
}

void VariantAdder::add_variants(vcflib::VariantCallFile* vcf) {
    
    // Make a buffer
    WindowedVcfBuffer buffer(vcf, variant_range);
    
    // Count how many variants we have done
    size_t variants_processed = 0;
    
    // Keep track of the previous contig name, so we know when to change our
    // progress bar.
    string prev_path_name;
    
    // We report when we skip contigs, but only once.
    set<string> skipped_contigs;
    
    while(buffer.next()) {
        // For each variant in its context of nonoverlapping variants
        vcflib::Variant* variant;
        vector<vcflib::Variant*> before;
        vector<vcflib::Variant*> after;
        tie(before, variant, after) = buffer.get_nonoverlapping();
        
        // Where is it?
        auto variant_path_name = vcf_to_fasta(variant->sequenceName);
        auto& variant_path_offset = variant->position; // Already made 0-based by the buffer
        
        if (!path_names.count(variant_path_name)) {
            // This variant isn't on a path we have.
            if (ignore_missing_contigs) {
                // That's OK. Just skip it.
                
                if (!skipped_contigs.count(variant_path_name)) {
                    // Warn first
                    
                    // Don't clobber an existing progress bar (which must be over since we must be on a new contig)
                    destroy_progress();
                    cerr << "warning:[vg::VariantAdder] skipping missing contig " << variant_path_name << endl;
                    skipped_contigs.insert(variant_path_name);
                }
                
                continue;
            } else {
                // Explode!
                throw runtime_error("Contig " + variant_path_name + " mentioned in VCF but not found in graph");
            }
        }
        
        if (variant->samples.empty()) {
            // Complain if the variant has no samples. If there are no samples
            // in the VCF, we can't generate any haplotypes to use to add the
            // variants.
            throw runtime_error("No samples in variant at " + variant_path_name +
                ":" + to_string(variant_path_offset) +"; can't make haplotypes");
        }
        
        // Grab the sequence of the path, which won't change
        const string& path_sequence = sync.get_path_sequence(variant_path_name);
    
        // Interlude: do the progress bar
        // TODO: not really thread safe
        if (variant_path_name != prev_path_name) {
            // Moved to a new contig
            prev_path_name = variant_path_name;
            destroy_progress();
            create_progress("contig " + variant_path_name, path_sequence.size());
        }
        update_progress(variant_path_offset);
        
        // Figure out what the actual bounds of this variant are. For big
        // deletions, the variant itself may be bigger than the window we're
        // using when looking for other local variants. For big insertions, we
        // might need to get a big subgraph to ensure we have all the existing
        // alts if they exist.
        
        // Make the list of all the local variants in one vector
        vector<vcflib::Variant*> local_variants = filter_local_variants(before, variant, after);
        
        // Where does the group of nearby variants start?
        size_t group_start = local_variants.front()->position;
        // And where does it end (exclusive)? This is the latest ending point of any variant in the group...
        size_t group_end = local_variants.back()->position + local_variants.back()->ref.size();
        
        // We need to make sure we also grab this much extra graph context,
        // since we count 2 radiuses + flank out from the ends of the group.
        size_t group_width = group_end - group_start;
        
        // Find the center and radius of the group of variants, so we know what graph part to grab.
        size_t overall_center;
        size_t overall_radius;
        tie(overall_center, overall_radius) = get_center_and_radius(local_variants);
        
        // Get the leading and trailing ref sequence on either side of this group of variants (to pin the outside variants down).

        // On the left we want either flank_range bases, or all the bases before
        // the first base in the group.
        size_t left_context_length = min((int64_t) flank_range, (int64_t) group_start);
        // On the right we want either flank_range bases, or all the bases after
        // the last base in the group. We know nothing will overlap the end of
        // the last variant, because we grabbed nonoverlapping variants.
        size_t right_context_length = min(path_sequence.size() - group_end, (size_t) flank_range);
    
        // Turn those into desired substring bounds.
        // TODO: this is sort of just undoing some math we already did
        size_t left_context_start = group_start - left_context_length;
        size_t right_context_past_end = group_end + right_context_length;
        
#ifdef debug
        cerr << "Original context bounds: " << left_context_start << " - " << right_context_past_end << endl;
#endif
        
        // Round bounds to node start and endpoints.
        sync.with_path_index(variant_path_name, [&](const PathIndex& index) {
            tie(left_context_start, right_context_past_end) = index.round_outward(left_context_start, right_context_past_end);
        });
        
#ifdef debug
        cerr << "New context bounds: " << left_context_start << " - " << right_context_past_end << endl;
#endif
        
        // Recalculate context lengths
        left_context_length = group_start - left_context_start;
        right_context_length = right_context_past_end - group_end;
        
        // Make sure we pull out out to the ends of the contexts
        overall_radius = max(overall_radius, max(overall_center - left_context_start, right_context_past_end - overall_center));
        
        // Get actual context strings
        string left_context = path_sequence.substr(group_start - left_context_length, left_context_length);
        string right_context = path_sequence.substr(group_end, right_context_length);
        
        // Get the unique haplotypes
        auto haplotypes = get_unique_haplotypes(local_variants, &buffer);
        
        // Track the total bp of haplotypes
        size_t total_haplotype_bases = 0;
        
        // Track the total graph size for the alignments
        size_t total_graph_bases = 0;
        
#ifdef debug
        cerr << "Have " << haplotypes.size() << " haplotypes for variant "
            << variant->sequenceName << ":" << variant->position << endl;
#endif
        
        for (auto& haplotype : haplotypes) {
            // For each haplotype
            
            // TODO: since we lock repeatedly, neighboring variants will come in
            // in undefined order and our result is nondeterministic.
            
            // Only look at haplotypes that aren't pure reference.
            bool has_nonreference = false;
            for (auto& allele : haplotype) {
                if (allele != 0) {
                    has_nonreference = true;
                    break;
                }
            }
            if (!has_nonreference) {
                // Don't bother aligning all-ref haplotypes to the graph.
                // They're there already.
#ifdef debug
                cerr << "Skip all-reference haplotype." << endl;
#endif
                continue;
            }
            
#ifdef debug
            cerr << "Haplotype ";
            for (auto& allele_number : haplotype) {
                cerr << allele_number << " ";
            }
            cerr << endl;
#endif
            
            // Make its combined string
            stringstream to_align;
            to_align << left_context << haplotype_to_string(haplotype, local_variants) << right_context;
            
#ifdef debug
            cerr << "Align " << to_align.str() << endl;
#endif

            // Count all the bases
            total_haplotype_bases += to_align.str().size();
            
            // Make a request to lock the subgraph, leaving the nodes we rounded
            // to (or the child nodes they got broken into) as heads/tails.
            GraphSynchronizer::Lock lock(sync, variant_path_name, left_context_start, right_context_past_end);
            
#ifdef debug
            cerr << "Waiting for lock on " << variant_path_name << ":"
                << left_context_start << "-" << right_context_past_end << endl;
#endif
            
            // Block until we get it
            lock_guard<GraphSynchronizer::Lock> guard(lock);
            
#ifdef debug
            cerr << "Got lock on " << variant_path_name << ":"
                << left_context_start << "-" << right_context_past_end << endl;
#endif            
                
#ifdef debug
            cerr << "Got " << lock.get_subgraph().length() << " bp in " << lock.get_subgraph().size() << " nodes" << endl;
#endif

#ifdef debug
            ofstream seq_dump("seq_dump.txt");
            seq_dump << to_align.str();
            seq_dump.close();

            sync.with_path_index(variant_path_name, [&](const PathIndex& index) {
                // Make sure we actually have the endpoints we wanted
                auto found_left = index.find_position(left_context_start);
                auto found_right = index.find_position(right_context_past_end - 1);
                assert(left_context_start == found_left->first);
                assert(right_context_past_end == found_right->first + index.node_length(found_right));
                
                cerr << "Group runs " << group_start << "-" << group_end << endl;
                cerr << "Context runs " << left_context_start << "-" << right_context_past_end << ": "
                    << right_context_past_end - left_context_start  << " bp" << endl;
                cerr << "Sequence is " << to_align.str().size() << " bp" << endl;
                cerr << "Leftmost node is " << found_left->second << endl;
                cerr << "Leftmost Sequence: " << lock.get_subgraph().get_node(found_left->second.node)->sequence() << endl;
                cerr << "Rightmost node is " << found_right->second << endl;
                cerr << "Rightmost Sequence: " << lock.get_subgraph().get_node(found_right->second.node)->sequence() << endl;
                cerr << "Left context: " << left_context << endl;
                cerr << "Right context: " << right_context << endl;
                
                lock.get_subgraph().for_each_node([&](Node* node) {
                    // Look at nodes
                    if (index.by_id.count(node->id())) {
                        cerr << "Node " << node->id() << " at " << index.by_id.at(node->id()).first
                            << " orientation " << index.by_id.at(node->id()).second << endl;
                    } else {
                        cerr << "Node " << node->id() << " not on path" << endl;
                    }
                });
                
                if (lock.get_subgraph().is_acyclic()) {
                    cerr << "Subgraph is acyclic" << endl;
                } else {
                    cerr << "Subgraph is cyclic" << endl;
                }
            });
#endif
            
            // Work out how far we would have to unroll the graph to account for
            // a giant deletion. We also want to account for alts that may
            // already be in the graph and need unrolling for a long insert.
            size_t max_span = max(right_context_past_end - left_context_start, to_align.str().size());
            
            // Record the size of graph we're aligning to in bases
            total_graph_bases += lock.get_subgraph().length();
            
            // Do the alignment, dispatching cleverly on size
            Alignment aln = smart_align(lock.get_subgraph(), lock.get_endpoints(), to_align.str(), max_span);
            
            if (local_variants.size() == 1) {
                // We only have one variant here, so we ought to have at least the expected giant gap score
                
                // Calculate the expected min score, for a giant gap (i.e. this
                // variant is an SV indel), assuming a perfect match context
                // background.
                int64_t min_length = min(right_context_past_end - left_context_start, to_align.str().size());
                int64_t max_length = max(right_context_past_end - left_context_start, to_align.str().size());
                int64_t expected_score = min_length * aligner.match -
                    aligner.gap_open - 
                    (max_length - 1 - min_length) * aligner.gap_extension;
                    
                // But maybe we don't have a massive indel and really have just
                // a SNP or something. We should accept any positive scoring
                // alignment.
                expected_score = min(expected_score, (int64_t) 0);
                
                assert(aln.score() >= expected_score);
            }
            
            // We shouldn't have dangling ends, really, but it's possible for
            // inserts that have copies already in the graph to end up producing
            // alignments just as good as the alignment we wanted that have
            // their gaps pushed to one end or the other, and we need to
            // tolerate them and make their insertions.
            
            // We know the aligner left-shifts the gaps for inserts, so make
            // sure that we at least *end* with a match.
            assert(aln.path().mapping_size() > 0);
            auto& last_mapping = aln.path().mapping(aln.path().mapping_size() - 1);
            assert(last_mapping.edit_size() > 0);
            auto& last_edit = last_mapping.edit(last_mapping.edit_size() - 1);
            assert(edit_is_match(last_edit));
            
            // Find the first edit and get its oriented node
            auto& first_mapping = aln.path().mapping(0);
            assert(first_mapping.edit_size() > 0);
            auto& first_edit = first_mapping.edit(0);
            
            // Construct the NodeSide on the left of the graph in the orientation the graph is aligned to.
            NodeSide left_of_alignment(first_mapping.position().node_id(), first_mapping.position().is_reverse());
            
            // Get all the NodeSides connected to it in the periphery of the
            // graph we extracted.
            set<NodeSide> connected = lock.get_peripheral_attachments(left_of_alignment);
            
#ifdef debug
            cerr << "Alignment starts at " << left_of_alignment << " which connects to ";
            for (auto& c : connected) {
                cerr << c << ", ";
            }
            cerr << endl;
#endif
            
            // Make this path's edits to the original graph. We don't need to do
            // anything with the translations. Handle insertions on the very
            // left by attaching them to whatever is attached to our leading
            // node.
            lock.apply_edit(aln.path(), connected);

        }
        
        if (variants_processed++ % 1000 == 0 || true) {
            #pragma omp critical (cerr)
            cerr << "Variant " << variants_processed << ": " << haplotypes.size() << " haplotypes at "
                << variant->sequenceName << ":" << variant->position << ": "
                << (total_haplotype_bases / haplotypes.size()) << " bp vs. "
                << (total_graph_bases / haplotypes.size()) << " bp haplotypes vs. graphs average" << endl;
        }
        
    }

    // Clean up after the last contig.
    destroy_progress();
    
}

Alignment VariantAdder::smart_align(vg::VG& graph, pair<NodeSide, NodeSide> endpoints, const string& to_align, size_t max_span) {

    // We need this fro reverse compelmenting alignments
    auto node_length_function = [&](id_t id) {
        return graph.get_node(id)->sequence().size();
    };

    // Decide what kind of alignment we need to do. Whatever we pick,
    // we'll fill this in.
    Alignment aln;
    
    if (to_align.size() <= whole_alignment_cutoff && graph.length() < whole_alignment_cutoff) {
        // If the graph and the string are short, do a normal banded global
        // aligner with permissive banding and the whole string length as
        // band padding. We can be inefficient but we won't bring down the
        // system.

        cerr << "Attempt full-scale " << to_align.size() << " x " << graph.length() << " alignment" << endl;
        
        // Do the alignment in both orientations
        
        // Align in the forward orientation using banded global aligner, unrolling for large deletions.
        aln = graph.align(to_align, &aligner, 0, false, false, 0, true, 0, max_span);
        // Align in the reverse orientation using banded global aligner, unrolling for large deletions.
        // TODO: figure out which way our reference path goes through our subgraph and do half the work.
        // Note that if we have reversing edges and a lot of unrolling, we might get the same alignment either way.
        Alignment aln2 = graph.align(reverse_complement(to_align), &aligner,
            0, false, false, 0, true, 0, max_span);
        
        // Note that the banded global aligner doesn't fill in identity.
        
#ifdef debug
        cerr << "Scores: " << aln.score() << " fwd vs. " << aln2.score() << " rev" << endl;
#endif
            
        if (aln2.score() > aln.score()) {
            // The reverse alignment is better. But spit it back in the
            // forward orientation.
            aln = reverse_complement_alignment(aln2, node_length_function);
        }

#ifdef debug
        cerr << "Subgraph: " << pb2json(graph.graph) << endl;            
        cerr << "Alignment: " << pb2json(aln) << endl;
#endif
        
    } else {
        // Either the graph or the sequence to align is too big to just
        // throw in to the banded aligner with big bands.
        
        // First try the endpoint alignments and see if they look like the whole thing might be in the graph.
        
        
        // We need to figure out what bits we'll align
        string left_tail; 
        string right_tail;
        
        if (to_align.size() <= pinned_tail_size) {
            // Each tail will just be the whole string
            left_tail = to_align;
            right_tail = to_align;
        } else {
            // Cut off the tails
            left_tail = to_align.substr(0, pinned_tail_size);
            right_tail = to_align.substr(to_align.size() - pinned_tail_size);
        }
        
        // We don't want to try to align against truly massive graphs with
        // gssw because we can overflow. We also know our alignments need to
        // be near the ends of the extracted graph, so there's no point
        // aligning to the middle.
        
        // Extract one subgraph at each end of the big subgraph we're
        // aligning to. Since we know where we extracted the original
        // subgraph from, this is possible.
        VG left_subgraph;
        VG right_subgraph;
        left_subgraph.add_node(*graph.get_node(endpoints.first.node));
        right_subgraph.add_node(*graph.get_node(endpoints.second.node));
        graph.expand_context_by_length(left_subgraph, left_tail.size() * 2);
        graph.expand_context_by_length(right_subgraph, right_tail.size() * 2);
        
        cerr << "Attempt two smaller " << left_tail.size() << " x " << left_subgraph.length()
            << " and " << right_tail.size() << " x " << right_subgraph.length() << " alignments" << endl;
        
        // Do the two pinned tail alignments on the forward strand, pinning
        // opposite ends.
        Alignment aln_left = left_subgraph.align(left_tail, &aligner,
            0, true, true, 0, false, 0, max_span);
        Alignment aln_right = right_subgraph.align(right_tail, &aligner,
            0, true, false, 0, false, 0, max_span);
            
        if (aln_left.path().mapping_size() < 1 ||
            aln_left.path().mapping(0).position().node_id() != endpoints.first.node ||
            aln_left.path().mapping(0).edit_size() < 1 || 
            !edit_is_match(aln_left.path().mapping(0).edit(0))) {
        
            // The left alignment didn't start with a match to the correct
            // endpoint node. Try aligning it in reverse complement, and
            // pinning the other end.
            
            // TODO: what if we have an exact palindrome over a reversing
            // edge, and we keep getting the same alignment arbitrarily no
            // matter how we flip the sequence to align?
            aln_left = reverse_complement_alignment(left_subgraph.align(reverse_complement(left_tail), &aligner,
                0, true, false, 0, false, 0, max_span), node_length_function);
                
        }
        
        // It's harder to do the same checks on the right side because we
        // can't just look at 0. Go find the rightmost mapping and edit, if
        // any.
        const Mapping* rightmost_mapping = nullptr;
        const Edit* rightmost_edit = nullptr;
        if (aln_right.path().mapping_size() > 0) {
            rightmost_mapping = &aln_right.path().mapping(aln_right.path().mapping_size() - 1);
        }
        if (rightmost_mapping != nullptr && rightmost_mapping->edit_size() > 0) {
            rightmost_edit = &rightmost_mapping->edit(rightmost_mapping->edit_size() - 1);
        }
        
        if (rightmost_mapping == nullptr ||
            rightmost_mapping->position().node_id() != endpoints.second.node ||
            rightmost_edit == nullptr || 
            !edit_is_match(*rightmost_edit)) {
        
            // The right alignment didn't end with a match to the correct
            // endpoint node. Try aligning it in reverse complement and
            // pinning the other end.
            
            // TODO: what if we have an exact palindrome over a reversing
            // edge, and we keep getting the same alignment arbitrarily no
            // matter how we flip the sequence to align?
            aln_right = reverse_complement_alignment(right_subgraph.align(reverse_complement(right_tail), &aligner,
                0, true, true, 0, false, 0, max_span), node_length_function);
        }
        
        cerr << "\tScores: " << aln_left.score() << "/" << left_tail.size() * aligner.match * min_score_factor
            << ", " << aln_right.score() << "/" << right_tail.size() * aligner.match * min_score_factor << endl;
        
        if (aln_left.score() > left_tail.size() * aligner.match * min_score_factor ||
            aln_right.score() > right_tail.size() * aligner.match * min_score_factor) {
        
            // Aligning the two tails suggests that the whole string might be in
            // the graph already.
            
            if (false) {
                // It's safe to try the tight banded alignment
        
                // We set this to true if we manage to find a valid alignment in the
                // narrow band.
                bool aligned_in_band;
                
                try {
                
                    cerr << "Attempt thin " << to_align.size() << " x " << graph.length() << " alignment" << endl;
                
                    // Throw it into the aligner with very restrictive banding to see if it's already basically present
                    aln = graph.align(to_align, &aligner,
                        0, false, false, 0, true, large_alignment_band_padding, max_span);
                    Alignment aln2 = graph.align(reverse_complement(to_align), &aligner,
                        0, false, false, 0, true, large_alignment_band_padding, max_span);
                    if (aln2.score() > aln.score()) {
                        // The reverse alignment is better. But spit it back in the
                        // forward orientation.
                        aln = reverse_complement_alignment(aln2, node_length_function);
                    }
                    aligned_in_band = true;
                } catch(NoAlignmentInBandException ex) {
                    // If the aligner can't find any valid alignment in the restrictive
                    // band, we will need to knock together an alignment manually.
                    aligned_in_band = false;
                }
                
                if (aligned_in_band && aln.score() > to_align.size() * aligner.match * min_score_factor) {
                    // If we get a good score, use that alignment
#ifdef debug
                    cerr << "Found sufficiently good restricted banded alignment" << endl;
#endif
                    return aln;
                }
                
            } else {
            
                cerr << "Attempt mapper-based " << to_align.size() << " x " << graph.length() << " alignment" << endl;
            
                // Otherwise, it's unsafe to try the tight banded alignment
                // (because our bands might get too big). Try a Mapper-based
                // fake-banded alignment, and trturn its alignment if it finds a
                // good one.
                
                // Generate an XG index
                xg::XG xg_index(graph.graph);

                // Generate a GCSA2 index
                gcsa::GCSA* gcsa_index = nullptr;
                gcsa::LCPArray* lcp_index = nullptr;
    
                if (edge_max) {
                    VG gcsa_graph = graph; // copy the graph
                    // remove complex components
                    gcsa_graph.prune_complex_with_head_tail(kmer_size, edge_max);
                    if (subgraph_prune) gcsa_graph.prune_short_subgraphs(subgraph_prune);
                    // then index
                    cerr << "\tGCSA index size: " << gcsa_graph.length() << " bp" << endl;
                    gcsa_graph.build_gcsa_lcp(gcsa_index, lcp_index, kmer_size, false, false, doubling_steps);
                } else {
                    // if no complexity reduction is requested, just build the index
                    cerr << "\tGCSA index size: " << graph.length() << " bp" << endl;
                    graph.build_gcsa_lcp(gcsa_index, lcp_index, kmer_size, false, false, doubling_steps);
                }
                        
                // Make the Mapper
                Mapper mapper(&xg_index, gcsa_index, lcp_index);
                // Copy over alignment scores
                mapper.set_alignment_scores(aligner.match, aligner.mismatch, aligner.gap_open, aligner.gap_extension);
                
                // Map. Will invoke the banded aligner if the read is long, and
                // the normal index-based aligner otherwise.
                // Note: reverse complement is handled by the mapper.
                aln = mapper.align(to_align);
                
                // Clean up indexes
                delete lcp_index;
                delete gcsa_index;
                
                cerr << "\tScore: " << aln.score() << endl;
                
                if (aln.score() > to_align.size() * aligner.match * min_score_factor) {
                    // This alignment looks good.
                    
                    // TODO: check for banded alignments that jump around and
                    // have lots of matches but also lots of novel
                    // edges/deletions
                    
                    return aln;
                }
                
                
            }
        
        }
        
        // If we get here, we couldn't find a good banded alignment, or it looks like the ends aren't present already.
        cerr << "Splicing tail alignments" << endl;
        
        // Splice left and right tails together with any remaining sequence we didn't have
        
        // How much overlap is there between the two tails? May be negative.
        int64_t overlap = (int64_t) aln_left.sequence().size() +
            (int64_t) aln_right.sequence().size() - (int64_t) to_align.size();
        
        if (overlap >= 0) {
            // All of the string is accounted for in these two
            // alignments, and we can cut them and splice them.
            
            // Take half the overlap off each alignment and paste together
            aln = simplify(merge_alignments(strip_from_end(aln_left, overlap / 2),
                strip_from_start(aln_right, (overlap + 1) / 2)));
                
            // TODO: produce a better score!
            aln.set_score(aln_left.score() + aln_right.score());
            
#ifdef debug
            cerr << "Spliced overlapping end alignments" << endl;
#endif
            
        } else {
            // Not all of the string is accounted for in these two
            // alignments, so we will splice them together with any
            // remaining input sequence.
            
            string middle_sequence = to_align.substr(aln_left.sequence().size(), -overlap);
            
            // Make a big bogus alignment with an unplaced pure insert mapping.
            Alignment aln_middle;
            aln_middle.set_sequence(middle_sequence);
            auto* middle_mapping = aln_middle.mutable_path()->add_mapping();
            auto* middle_edit = middle_mapping->add_edit();
            middle_edit->set_sequence(middle_sequence);
            middle_edit->set_to_length(middle_sequence.size());
            
            // Paste everything together
            aln = simplify(merge_alignments(merge_alignments(aln_left, aln_middle), aln_right));
            
            // TODO: produce a better score!
            aln.set_score(aln_left.score() + aln_right.score());
            
#ifdef debug
            cerr << "Spliced disconnected end alignments" << endl;
#endif
            
        }
            
    }
    
    // TODO: check if we got alignments that didn't respect our specified
    // endpoints by one of the non-splicing-together alignment methods.
    
    return aln;

}


set<vector<int>> VariantAdder::get_unique_haplotypes(const vector<vcflib::Variant*>& variants, WindowedVcfBuffer* cache) const {
    set<vector<int>> haplotypes;
    
    if (variants.empty()) {
        // Nothing's there
        return haplotypes;
    }
    
    for (size_t sample_index = 0; sample_index < variants.front()->sampleNames.size(); sample_index++) {
        // For every sample
        auto& sample_name = variants.front()->sampleNames[sample_index];
        
        // Make its haplotype(s) on the region. We have a map from haplotype
        // number to actual vector. We'll tack stuff on the ends when they are
        // used, then throw out any that aren't full-length.
        map<size_t, vector<int>> sample_haplotypes;
        
        
        for (auto* variant : variants) {
            // Get the genotype for each sample
            const vector<int>* genotype;
            
            if (cache != nullptr) {
                // Use the cache provided by the buffer
                genotype = &cache->get_parsed_genotypes(variant).at(sample_index);
            } else {
                // Parse from the variant ourselves
                auto genotype_string = variant->getGenotype(sample_name);
            
                // Fake it being phased
                replace(genotype_string.begin(), genotype_string.end(), '/', '|');
                
                genotype = new vector<int>(vcflib::decomposePhasedGenotype(genotype_string));
            }
            
#ifdef debug
            cerr << "Genotype of " << sample_name << " at " << variant->position << ": ";
            for (auto& alt : *genotype) {
                cerr << alt << " ";
            }
            cerr << endl;
#endif
            
            for (size_t phase = 0; phase < genotype->size(); phase++) {
                // For each phase in the genotype
                
                // Get the allele number and ignore missing data
                int allele_index = (*genotype)[phase];
                if (allele_index == vcflib::NULL_ALLELE) {
                    allele_index = 0;
                }
                
                if (allele_index >= variant->alleles.size()) {
                    // This VCF has out-of-range alleles
                    //cerr << "error:[vg::VariantAdder] variant " << variant->sequenceName << ":" << variant->position
                    //    << " has invalid allele index " << allele_index
                    //    << " but only " << variant->alt.size() << " alts" << endl;
                    // TODO: right now we skip them as a hack
                    allele_index = 0;
                    //exit(1);
                }
                
                // Stick each allele number at the end of its appropriate phase
                sample_haplotypes[phase].push_back(allele_index);
            }
            
            if (cache == nullptr) {
                // We're responsible for this vector
                delete genotype;
            }
        }
        
        for (auto& kv : sample_haplotypes) {
            auto& haplotype = kv.second;
            // For every haplotype in this sample
            if (haplotype.size() != variants.size()) {
                // If it's not the full length, it means some variants don't
                // have it. Skip.
                continue;
            }
            
            // Otherwise, add it to the set of observed haplotypes
            haplotypes.insert(haplotype);
        }
    }
    
    // After processing all the samples, return the unique haplotypes
    return haplotypes;
    
}

string VariantAdder::haplotype_to_string(const vector<int>& haplotype, const vector<vcflib::Variant*>& variants) {
    // We'll fill this in with variants and separating sequences.
    stringstream result;
    
    // These lists need to be in 1 to 1 correspondence
    assert(haplotype.size() == variants.size());
    
    if (variants.empty()) {
        // No variants means no string representation.
        return "";
    }
    
    // Do the first variant
    result << variants.front()->alleles.at(haplotype.at(0));
    
    for (size_t i = 1; i < variants.size(); i++) {
        // For each subsequent variant
        auto* variant = variants.at(i);
        auto* last_variant = variants.at(i - 1);
        
        // Do the intervening sequence.
        // Where does that sequence start?
        size_t sep_start = last_variant->position + last_variant->ref.size();
        // And how long does it run?
        size_t sep_length = variant->position - sep_start;
        
        // Pull out the separator sequence and tack it on.
        result << sync.get_path_sequence(vcf_to_fasta(variant->sequenceName)).substr(sep_start, sep_length);

        // Then put the appropriate allele of this variant.
        result << variant->alleles.at(haplotype.at(i));
    }
    
    return result.str();
}

size_t VariantAdder::get_radius(const vcflib::Variant& variant) {
    // How long is the longest alt?
    size_t longest_alt_length = variant.ref.size();
    for (auto& alt : variant.alt) {
        // Take the length of the longest alt you find
        longest_alt_length = max(longest_alt_length, alt.size());
    }
    
    // Report half its length, and don't round down.
    return (longest_alt_length + 1) / 2;
}


size_t VariantAdder::get_center(const vcflib::Variant& variant) {
    // Where is the end of the variant in the reference?
    size_t path_last = variant.position + variant.ref.size() - 1;
    
    // Where is the center of the variant in the reference?
    return (variant.position + path_last) / 2;
}


pair<size_t, size_t> VariantAdder::get_center_and_radius(const vector<vcflib::Variant*>& variants) {

    // We keep track of the leftmost and rightmost positions we would need to
    // cover, which may be negative on the left.
    int64_t leftmost = numeric_limits<int64_t>::max();
    int64_t rightmost = 0;

    for (auto* v : variants) {
        // For every variant
        auto& variant = *v;
        
        // Work out its center (guaranteed positive)
        int64_t center = get_center(variant);
        // And its radius
        int64_t radius = get_radius(variant);
        
        // Expand the range of the block if needed
        leftmost = min(leftmost, center - radius);
        rightmost = max(rightmost, center + radius);
    }
    
    // Calculate the center between the two ends, and the radius needed to hit
    // both ends.
    size_t overall_center = (leftmost + rightmost) / 2;
    size_t overall_radius = (rightmost - leftmost + 1) / 2;
    
    return make_pair(overall_center, overall_radius);

}

vector<vcflib::Variant*> VariantAdder::filter_local_variants(const vector<vcflib::Variant*>& before,
    vcflib::Variant* variant, const vector<vcflib::Variant*>& after) const {

    // This is the filter we apply
    auto filter = [&](vcflib::Variant* v) {
        // Keep a variant if it isn't too big
        return get_radius(*v) <= max_context_radius;
    };

    // Make the list of all the local variants in one vector
    vector<vcflib::Variant*> local_variants;
    
    // Keep the nearby variants if they pass the test
    copy_if(before.begin(), before.end(), back_inserter(local_variants), filter);
    // And the main variant always
    local_variants.push_back(variant);
    copy_if(after.begin(), after.end(), back_inserter(local_variants), filter);

#ifdef debug
        cerr << "Local variants: ";
        for (auto* v : local_variants) {
            cerr << vcf_to_fasta(v->sequenceName) << ":" << v->position << " ";
        }
        cerr << endl;
#endif

    return local_variants;
}

}





















